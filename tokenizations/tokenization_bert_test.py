import os
from tokenizations import tokenization_bert

vocab_file_path = os.path.join(os.path.dirname(__file__), '../cache/vocab_all.txt')
tokenizer = tokenization_bert.BertTokenizer(vocab_file=vocab_file_path)

test_term = '[MASK] 长忆西山。 [SEP] [CLS] '  # 注意特殊token之间需要添加空格，避免按中文分字切token的时候混淆

tokens = tokenizer.tokenize(test_term)
ids = tokenizer.convert_tokens_to_ids(tokens)

print(tokens)
print(ids)
