import os
import pickle

# Classes and types from dataset

# 17 classes
FULL_CLEANED_CLASSES = [
    'Camel', 'HIS_mouse', 'human', 'mouse_BALB_c', 'mouse_Balb_c', 'mouse_C57BL_6',
    'mouse_C57BL_6J', 'mouse_Igh_wt', 'mouse_Ighe_e', 'mouse_Ighg_g',
    'mouse_RAG2_GFP_129Sve', 'mouse_Swiss_Webster', 'mouse_outbred',
    'mouse_outbred_C57BL_6', 'rabbit', 'rat', 'rhesus',
]

# 2 typees
FULL_CLEANED_TYPES = [
    'Heavy', 'Light'
]

unique_classes = set(FULL_CLEANED_CLASSES)
unique_types = set(FULL_CLEANED_TYPES)

# Create Vocab

# dictionary of all tokens
amino_acids = "ARNDCQEGHILKMFPSTWYV"
base_special = ['<pad>','<eos>', '<unk>', '<sos>'] 
class_tokens = [f'<cls_{c}>' for c in unique_classes if c]
type_tokens = [f'<type_{t}>' for t in unique_types if t]

special_tokens = base_special + class_tokens + type_tokens + ["-"] 
chars = list(amino_acids) + special_tokens
chars = sorted(list(set(chars))) 
stoi = {ch: i for i, ch in enumerate(chars)} 
itos = {i: ch for i, ch in enumerate(chars)} 
vocab_size = len(chars)

print(f"Vocab size (full, including '-'): {vocab_size}")

# Removed Processing

if __name__ == '__main__':
    
    output_dir = os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)
    
    NUM_TRAIN = 203968932
    NUM_VAL = 11819793     

    print("\nGenerating meta.pkl...")
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'num_train': NUM_TRAIN,
        'num_val': NUM_VAL,
    }

    meta_path = os.path.join(output_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    print(f"\nReady. Generated {meta_path}.")