import os
import pickle
import numpy as np
from datasets import load_dataset
from collections import Counter

print("Loading dataset...")

# 1. Загрука набора данных OAS95-aligned-cleaned
dataset = load_dataset("bayes-group-diffusion/OAS95-aligned-cleaned", split="train", streaming=True)
sequences = []
unique_classes = set()
unique_types = set()
print("Start download sequences...")
for idx, item in enumerate(dataset):
    if idx >= 10000: break
    seq = item.get('sequence', '')
    if seq:
        cls = item.get('class', '').replace("/", "_").replace(" ", "")
        typ = item.get('type', '')
        unique_classes.add(cls)
        unique_types.add(typ)
        sequences.append({
            'sequence': seq.upper(),
            'init_seq': item.get('init_seq', '').upper(),
            'class': cls,  # очищенный
            'type': typ
        })

print(f"Downloaded {len(sequences)} sequences.")
print(f"Uniq class: {len(unique_classes)}")
print(f"Uniq type: {len(unique_types)}")

# 2. Создание словаря аминокислот (20 видов + 5 префиксов)
amino_acids = "ARNDCQEGHILKMFPSTWYV"
base_special = ['<pad>', '<eos>', '<unk>', '<sos>']
class_tokens = [f'<cls_{c}>' for c in unique_classes if c]
type_tokens = [f'<type_{t}>' for t in unique_types if t]

special_tokens = base_special + class_tokens + type_tokens
chars = list(amino_acids) + special_tokens
chars = sorted(list(set(chars)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

print(f"Vocab size: {vocab_size}")

with open('input/sequences.pkl', 'wb') as f:
    pickle.dump(sequences, f)

print("Encoding sequences...")

meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
    'num_sequences': len(sequences),
    'avg_length': np.mean([len(s['sequence']) for s in sequences])
}

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Ready. Files: sequences.pkl, input/meta.pkl")
