import os
import pickle
import numpy as np
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm


print("Downloading train split...")
train_dataset = load_dataset("bayes-group-diffusion/OAS95-aligned-cleaned", split="train", streaming=True)
train_sequences = []
unique_classes = set()
unique_types = set()

print("Start download sequences...")
for idx, item in tqdm(enumerate(train_dataset), desc="Loading test sequences"):
    if idx >= 2000000: break
    seq = item.get('sequence', '')
    if seq:
        cls = item.get('class', '').replace("/", "_").replace(" ", "").replace("-","_")
        typ = item.get('type', '')
        unique_classes.add(cls)
        unique_types.add(typ)
        train_sequences.append({
            'sequence': seq.upper(),
            'init_seq': item.get('init_seq', '').upper(),
            'class': cls,  # очищенный
            'type': typ
        })

test_dataset = load_dataset("bayes-group-diffusion/OAS95-aligned-cleaned", split="test", streaming=True)
test_sequences = []

print("Downloading test split...")
for idx, item in tqdm(enumerate(test_dataset), desc="Loading test sequences"):
    # if idx >= 10000: 
    #     break
    seq = item.get('sequence', '')
    if seq:
        cls = item.get('class', '').replace("/", "_").replace(" ", "").replace("-","_")
        typ = item.get('type', '')
        unique_classes.add(cls)
        unique_types.add(typ)
        test_sequences.append({
            'sequence': seq.upper(),
            'init_seq': item.get('init_seq', '').upper(),
            'class': cls,
            'type': typ
        })

print(f"Train: {len(train_sequences)} sequences")
print(f"Test (val): {len(test_sequences)} sequences")
print(f"Unique classes: {len(unique_classes)}")
print(f"Unique types: {len(unique_types)}")

# 2. Создание словаря аминокислот (20 видов + 5 префиксов)
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

print(f"Vocab size: {vocab_size}")

# with open(os.path.join(os.path.dirname(__file__), 'sequences.pkl'), 'wb') as f:
#     pickle.dump(sequences, f)

print("Saving sequences to files")
with open(os.path.join(os.path.dirname(__file__), 'train_sequences.pkl'), 'wb') as f:
    pickle.dump(train_sequences, f)

with open(os.path.join(os.path.dirname(__file__), 'test_sequences.pkl'), 'wb') as f:
    pickle.dump(test_sequences, f)

print("Generatin meta...")
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
    'num_train': len(train_sequences),
    'num_val': len(test_sequences),
    'avg_train_length': np.mean([len(s['sequence']) for s in train_sequences]) if train_sequences else 0,
    'avg_val_length': np.mean([len(s['sequence']) for s in test_sequences]) if test_sequences else 0
}

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Ready. Files: train_sequences.pkl, test_sequences.pkl, meta.pkl")
