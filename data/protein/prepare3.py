# saves the OAS95-aligned-cleaned dataset to structured files using Hugging Face datasets.
# This script saves raw strings of sequences to allow dynamic tokenization and vocab filtering 
# based on 'sequence' vs 'init_seq' at training time.

import os
import pickle
import numpy as np
from tqdm import tqdm
from datasets import load_dataset 
import gc

# --- 1. Classes and types from dataset ---
FULL_CLEANED_CLASSES = [
    'Camel', 'HIS_mouse', 'human', 'mouse_BALB_c', 'mouse_Balb_c', 'mouse_C57BL_6',
    'mouse_C57BL_6J', 'mouse_Igh_wt', 'mouse_Ighe_e', 'mouse_Ighg_g',
    'mouse_RAG2_GFP_129Sve', 'mouse_Swiss_Webster', 'mouse_outbred',
    'mouse_outbred_C57BL_6', 'rabbit', 'rat', 'rhesus',
]

FULL_CLEANED_TYPES = [
    'Heavy', 'Light'
]

unique_classes = set(FULL_CLEANED_CLASSES)
unique_types = set(FULL_CLEANED_TYPES)

# --- 2. Create Vocab ---
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

def process_raw_batch(batch):
    """Process batch without holding everything in memory"""
    processed_sequences = []
    processed_init_seqs = []
    processed_classes = []
    processed_types = []
    processed_lens = []
    
    for i in range(len(batch['sequence'])):
        cls = batch['class'][i].replace("/", "_").replace(" ", "").replace("-","_")
        typ = batch['type'][i]
        seq = batch['sequence'][i].upper()
        init_seq = batch['init_seq'][i].upper()
        
        processed_sequences.append(seq)
        processed_init_seqs.append(init_seq)
        processed_classes.append(cls)
        processed_types.append(typ)
        processed_lens.append(len(seq))
    
    return {
        'sequence': processed_sequences,
        'init_seq': processed_init_seqs,
        'class_clean': processed_classes,
        'type_clean': processed_types,
        'len': processed_lens
    }

if __name__ == '__main__':
    # Уменьшаем количество процессов для экономии памяти
    num_proc = 2  # было 8
    num_proc_load_dataset = 1  # было 8
    
    print("\nStart loading from Hugging Face...")
    
    # Загружаем по частям
    train_dataset = load_dataset("bayes-group-diffusion/OAS95-aligned-cleaned", 
                               split="train[:1000]", 
                               num_proc=num_proc_load_dataset)
    test_dataset = load_dataset("bayes-group-diffusion/OAS95-aligned-cleaned", 
                              split="test", 
                              num_proc=num_proc_load_dataset)

    dataset = {
        'train': train_dataset,
        'val': test_dataset 
    }
    
    # Обрабатываем и сохраняем каждый сплит по отдельности
    output_dir = os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)
    
    processed_dataset = {}
    
    for split_name, dset in dataset.items():
        print(f"Processing {split_name} split...")
        
        # Обрабатываем батчами с очисткой памяти
        processed_dset = dset.map(
            process_raw_batch,
            batched=True,  # Обрабатываем батчами для эффективности
            batch_size=1000,  # Размер батча
            remove_columns=[col for col in dset.column_names if col not in ['sequence', 'init_seq', 'class', 'type']],
            desc=f"Processing {split_name}",
            num_proc=1,  # Уменьшаем до 1 процесса для стабильности
        )
        
        # Сохраняем сразу после обработки
        data_path = os.path.join(output_dir, f'{split_name}_hf_data')
        processed_dset.save_to_disk(data_path)
        print(f"Saved {split_name} data to: {data_path}")
        
        # Сохраняем ссылку и очищаем память
        processed_dataset[split_name] = processed_dset
        del processed_dset
        gc.collect()
        
    # --- 4. Save Meta ---
    print("\nGenerating meta.pkl...")
    
    # Вычисляем статистики без загрузки всех данных в память
    train_lens = [len(seq) for seq in processed_dataset['train']['sequence']]
    val_lens = [len(seq) for seq in processed_dataset['val']['sequence']] if len(processed_dataset['val']) else []
    
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'num_train': len(processed_dataset['train']),
        'num_val': len(processed_dataset['val']),
        'avg_train_length': np.mean(train_lens) if train_lens else 0,
        'avg_val_length': np.mean(val_lens) if val_lens else 0
    }

    meta_path = os.path.join(output_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    print(f"\nReady. Files: train_hf_data/, val_hf_data/, {meta_path}")