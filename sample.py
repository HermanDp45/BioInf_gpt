"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 400 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
class_label = None
type_label = None
max_protein_length=155
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
stoi = None
itos = None
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    # Используем правильную encode функцию для посимвольного кодирования
    encode = lambda s: [stoi.get(c, stoi['<unk>']) for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

start_tokens = []

if load_meta and stoi is not None:  # Only add special tokens if we have the custom vocabulary
    # Add start token
    if '<sos>' in stoi:
        start_tokens.append(stoi['<sos>'])
        print("Added <sos> token")
    else:
        print("Warning: <sos> token not found in vocabulary")
    
    # Add class token if specified
    if class_label is not None:
        class_token = f"<cls_{class_label.replace('/', '_').replace(' ', '').replace('-', '_')}>"
        if class_token in stoi:
            start_tokens.append(stoi[class_token])
            print(f"Added class token: {class_token}")
        else:
            print(f"Warning: Class token '{class_token}' not in vocabulary")
    
    # Add type token if specified
    if type_label is not None:
        type_token = f"<type_{type_label}>"
        if type_token in stoi:
            start_tokens.append(stoi[type_token])
            print(f"Added type token: {type_token}")
        else:
            print(f"Warning: Type token '{type_token}' not in vocabulary")
else:
    if class_label is not None or type_label is not None:
        print("Warning: class_label and type_label are ignored because no meta.pkl found")

# Encode the user's text prompt
start_ids = encode(start)
start_tokens.extend(start_ids)  # Combine special tokens with user text

x = (torch.tensor(start_tokens, dtype=torch.long, device=device)[None, ...])

print(f"Final prompt tokens: {len(start_tokens)}")
print(f"Starting generation with: {decode(start_tokens)}")

# post-processing

# clean all after first <eos>
def truncate_at_eos(tokens, eos_token='<eos>'):
    try:
        idx = tokens.index(eos_token)
        return tokens[:idx]
    except ValueError:
        return tokens  # если <eos> нет — оставить как есть

# clean all special tokens after prefix
def clean_special_tokens(tokens, allowed_prefix_tokens=None):
    if allowed_prefix_tokens is None:
        allowed_prefix_tokens = {'<sos>'}
    
    # Найти конец префикса (первый токен, не начинающийся с '<')
    body_start = 0
    for i, t in enumerate(tokens):
        if not t.startswith('<'):
            body_start = i
            break
    else:
        # вся последовательность — спецтокены
        return []

    prefix = tokens[:body_start]
    body = tokens[body_start:]

    # Оставить только аминокислоты и '-'
    cleaned_body = [t for t in body if t in "ARNDCQEGHILKMFPSTWYV-"]

    return prefix, cleaned_body

# prefix cleaner
def validate_and_dedup_prefix(tokens):
    prefix = []
    seen_cls = False
    seen_type = False
    for t in tokens:
        if t.startswith('<cls_'):
            if not seen_cls:
                prefix.append(t)
                seen_cls = True
        elif t.startswith('<type_'):
            if not seen_type:
                prefix.append(t)
                seen_type = True
        elif t == '<sos>':
            prefix.append(t)
        else:
            # тело последовательности — выходим
            break
    return prefix

def postprocess_generated_sequence(tokens, stoi, itos):
    seq_str = [itos[i] for i in tokens if i in itos]
    
    seq_str = truncate_at_eos(seq_str, '<eos>')
    
    prefix, cleaned_body = clean_special_tokens(seq_str)
    
    #cleaned_prefix = validate_and_dedup_prefix(prefix)
    
    #protein_seq = ''.join([t for t in seq_str if t in "ARNDCQEGHILKMFPSTWYV-"])
    
    #return (''.join(cleaned_prefix) + "".join(cleaned_body))[:152]
    return "".join(cleaned_body)[:max_protein_length]

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(postprocess_generated_sequence(y[0].tolist(), stoi, itos))
