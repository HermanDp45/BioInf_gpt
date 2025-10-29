import os
import pickle
import numpy as np
from datasets import load_dataset
from collections import Counter

os.makedirs('input', exist_ok=True)

print("Loading dataset...")

# 1. Загрука набора данных OAS95-aligned-cleaned
dataset = load_dataset("bayes-group-diffusion/OAS95-aligned-cleaned", split="train", streaming=True)

sequence = []
sequence_types = []

print("Processing sequences...")
for idx, item in enumerate(dataset):
    if idx >= 10000:
        break
    seq = item.get('init_seq', item.get('sequence', ''))
    if len(seq) > 0:
        sequence.append(seq.upper())
        sequence_types.append('sequence') # подправить

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
encoded_sequences = [encode(seq) for seq in sequence]

split = int(0.9 * len(encoded_sequences))
train_data = encoded_sequences[:split]
val_data = encoded_sequences[split:]

print(f"train sequences: {len(train_data)}")
print(f"val sequences: {len(val_data)}")

# 4. Export to bin files
train_data = np.array(train_data, dtype=np.uint16)
val_data = np.array(val_data, dtype=np.uint16)
train_data.tofile('input/train.bin')
val_data.tofile('input/val.bin')

# 5. Save metadata
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
    'num_sequences': len(encoded_sequences),
    'avg_length': sum([len(seq) for seq in encoded_sequences])/len(encoded_sequences)
}

with open('input/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)