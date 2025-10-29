import os
import pickle
import numpy as np
from datasets import load_dataset
from collections import Counter

print("Loading dataset...")

# 1. Загрука набора данных OAS95-aligned-cleaned
dataset = load_dataset("bayes-group-diffusion/OAS95-aligned-cleaned", split="train", streaming=True)
sequence = []

print("Processing sequences...")
for idx, item in enumerate(dataset):
    if idx >= 10000:
        break
    seq = item.get('init_seq', item.get('sequence', ''))
    if seq: sequence.append(seq.upper())

print(f"Processed {len(sequence)} sequences.")

# 2. Создание словаря аминокислот (20 видов + 5 префиксов)
amino_acids = "ARNDCQEGHILKMFPSTWYV"
special_tokens = ['<class>', '<type>', '<pad>', '<eos>', '<unk>']

chars = list(amino_acids) + special_tokens
vocab_size = len(chars)
stoi = {ch: i for i,ch in enumerate(chars)}
itos = {i: ch for i,ch in enumerate(chars)}

print(f"vocab_size: {vocab_size}")
print(f"amino_acids: {amino_acids}")
print(f"special_tokens: {special_tokens}")

# 3. Encode
def encode(sequence):
    return [stoi.get(ch, stoi['<unk>']) for ch in sequence]

print("Encoding sequences...")
encoded = [encode(s) for s in sequence]

split = int(0.9 * len(encoded))
train_data, val_data = encoded[:split], encoded[split:]

print(f"train sequences: {len(train_data)}")
print(f"val sequences: {len(val_data)}")

def flatten(seqs):
    flat = []
    for s in seqs: flat.extend(s)
    return np.array(flat, dtype=np.uint16)

train_ids = flatten(train_data)
val_ids = flatten(val_data)

# 4. Export to bin files
train_data = np.array(train_data, dtype=np.uint16)
val_data = np.array(val_data, dtype=np.uint16)
train_data.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_data.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# 5. Save metadata
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
    'num_sequences': len(encoded),
    'avg_length': sum([len(seq) for seq in encoded])/len(encoded)
}

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)