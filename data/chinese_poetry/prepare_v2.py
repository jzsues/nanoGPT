import os
import requests
import tiktoken
import numpy as np

# download the test chinese poetry dataset

input_file_path = os.path.join(os.path.dirname(__file__), 'test_chinese_poetry_v1.txt')

with open(input_file_path, 'r') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# encode with tiktoken gpt2 bpe
# enc = tiktoken.get_encoding("gpt2")
# train_ids = enc.encode_ordinary(train_data)
# val_ids = enc.encode_ordinary(val_data)
from tokenizations import tokenization_bert

vocab_file_path = os.path.join(os.path.dirname(__file__), '../../cache/vocab_all.txt')
tokenizer = tokenization_bert.BertTokenizer(vocab_file=vocab_file_path)
tokenizer.max_len = 999999
train_ids = tokenizer.encode(train_data)
val_ids = tokenizer.encode(val_data)
print(f"vocab size:{tokenizer.vocab_size}")
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
