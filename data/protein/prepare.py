# saves the OAS95-aligned-cleaned dataset to structured files using Hugging Face datasets.
# This script saves raw strings of sequences to allow dynamic tokenization and vocab filtering 
# based on 'sequence' vs 'init_seq' at training time.

import os
import pickle
import numpy as np
from tqdm import tqdm
from datasets import load_dataset 

# --- 1. Classes and types from dataset ---
# Using for creating vocab

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

# --- 2. Create Vocab ---

# Parametrs
num_proc = 8 
num_proc_load_dataset = num_proc

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

# --- 3. Process and save raw rows ---

def process_raw(example):
    """
    Save raw rows and meta
    """
    # get clear classes and types
    cls = example.get('class', '').replace("/", "_").replace(" ", "").replace("-","_")
    typ = example.get('type', '')
    
    # save in upper reg
    seq = example.get('sequence', '').upper()
    init_seq = example.get('init_seq', '').upper()
    
    return {
        'sequence': seq, 
        'init_seq': init_seq,
        'class_clean': cls, 
        'type_clean': typ,
        'len': len(seq)
    }

if __name__ == '__main__':
    # Dataset downloading
    print("\nStart loading from Hugging Face...")
    train_dataset = load_dataset("bayes-group-diffusion/OAS95-aligned-cleaned", split="train[:100000]", num_proc=num_proc_load_dataset)
    test_dataset = load_dataset("bayes-group-diffusion/OAS95-aligned-cleaned", split="test", num_proc=num_proc_load_dataset)

    dataset = {
        'train': train_dataset,
        'val': test_dataset 
    }
    
    # Dataset process: save rows and meta.pcl 
    processed_dataset = {}
    for split_name, dset in dataset.items():
        processed_dataset[split_name] = dset.map(
            process_raw,
            remove_columns=[col for col in dset.column_names if col not in ['sequence', 'init_seq', 'class', 'type']],
            desc=f"processing the {split_name} split",
            num_proc=num_proc,
        )

    # --- 4. Save to disk (HF format) ---

    output_dir = os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nSave to disk (HF format) (HF format)...")

    for split, dset in processed_dataset.items():
        data_path = os.path.join(output_dir, f'{split}_hf_data')
        # save to disk not RAM
        dset.save_to_disk(data_path)
        print(f"Saved {split} data to: {data_path}")
        
    # --- 5. Save Meta ---

    print("\Generating meta.pkl...")
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'num_train': len(processed_dataset['train']),
        'num_val': len(processed_dataset['val']),
        'avg_train_length': np.mean(processed_dataset['train']['len']) if len(processed_dataset['train']) else 0,
        'avg_val_length': np.mean(processed_dataset['val']['len']) if len(processed_dataset['val']) else 0
    }

    meta_path = os.path.join(output_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    print(f"\Ready. files: train_hf_data/, val_hf_data/, {meta_path}")