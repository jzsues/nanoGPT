import os
from tokenizations import tokenization_bert

vocab_file_path = os.path.join(os.path.dirname(__file__), '../cache/vocab_all.txt')
tokenizer = tokenization_bert.BertTokenizer(vocab_file=vocab_file_path)

block_size = 384
test_term = '[CLS] 长忆西山。 [SEP] '  # 注意特殊token之间需要添加空格，避免按中文分字切token的时候混淆

tokens = tokenizer.tokenize(test_term)
ids = tokenizer.encode(test_term, padding=True)

print(tokens)
print(ids)
