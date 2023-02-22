import os
import requests
import tiktoken
import numpy as np
import pickle

# download the test chinese poetry dataset

input_file_path = os.path.join(os.path.dirname(__file__), 'test_chinese_poetry_v2.txt')


def read_raw_file_lines(raw_data_path):
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        print(f'reading lines and add special tokens: [CLS] text [SEP] ')
        rlines = f.readlines()
        rlines = [' [CLS] ' + line.replace('\n', ' [SEP] ') for line in rlines]
    rlines_len = len(rlines)
    return rlines, rlines_len


min_length = 16
lines, lines_len = read_raw_file_lines(input_file_path)

split_pos = int(lines_len * 0.9)
train_lines = lines[:split_pos]
val_lines = lines[split_pos:]

# encode with tiktoken gpt2 bpe
# enc = tiktoken.get_encoding("gpt2")
# train_ids = enc.encode_ordinary(train_data)
# val_ids = enc.encode_ordinary(val_data)
from tokenizations import tokenization_bert

vocab_file_path = os.path.join(os.path.dirname(__file__), '../../cache/vocab_all.txt')
tokenizer = tokenization_bert.BertTokenizer(vocab_file=vocab_file_path)
tokenizer.max_len = 999999
train_tokens = [tokenizer.tokenize(line) for line in train_lines if len(line) > min_length]
val_tokens = [tokenizer.tokenize(line) for line in val_lines if len(line) > min_length]
train_token_ids = [tokenizer.convert_tokens_to_ids(line) for line in train_tokens]
val_token_ids = [tokenizer.convert_tokens_to_ids(line) for line in val_tokens]
train_ids = []
val_ids = []
for ids in train_token_ids:
    train_ids.extend(ids)

for ids in val_token_ids:
    val_ids.extend(ids)

print(f"vocab size:{tokenizer.vocab_size}")
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': tokenizer.vocab_size,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
