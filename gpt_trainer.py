import json
import torch
import random
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import regex as re


# Dataset class
class XiaohongshuDataset(Dataset):
    def __init__(self, keyword_list, article_list, tokenizer, max_length):
        # define variables
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        # iterate through the dataset
        for keyword, article in zip(keyword_list, article_list):
            # prepare the text
            prep_txt = f'<|startoftext|>Keywords: {keyword}\nArticle: {article}<|endoftext|>'
            # tokenize
            encodings_dict = tokenizer(prep_txt, truncation=True,
                                       max_length=max_length, padding="max_length")
            # append to list
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
            self.labels.append(torch.tensor(encodings_dict['input_ids']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]


# Data load function
def load_xiaohongshu_dataset(tokenizer, random_seed=1,
                             file_path="data/xiaohongshu/articles.csv"):
    # load dataset and sample 10k reviews.
    df = pd.read_csv(file_path, encoding='UTF-8')
    df.fillna('None')
    df = df[['desc', 'keywords']]
    df = df.sample(len(df), random_state=1)

    df = df[df['desc'].apply(lambda x: len(str(x)) > 10)]
    df = df[df['keywords'].apply(lambda x: len(str(x)) > 5)]

    # divide into test and train
    X_train, X_test, y_train, y_test = \
        train_test_split(df['keywords'].tolist(), df['desc'].tolist(),
                         shuffle=True, test_size=0.05, random_state=random_seed)

    # get max length
    max_length_train = max([len(tokenizer.encode(text)) for text in X_train])
    max_length_test = max([len(tokenizer.encode(text)) for text in X_test])
    max_length = max([max_length_train, max_length_test]) + 10  # for special tokens (sos and eos) and fillers
    max_length = max(max_length, 300)
    print(f"Setting max length as {max_length}")

    # format into XiaohongshuDataset class
    train_dataset = XiaohongshuDataset(X_train, y_train, tokenizer, max_length=max_length)

    # return
    return train_dataset, (X_test, y_test)


# import
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPTNeoForCausalLM

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model
model_name = "EleutherAI/gpt-neo-125M"
seed = 42

# seed
torch.manual_seed(seed)

# iterate for N trials
for trial_no in range(3):

    print("Loading model...")
    # load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, bos_token='<|startoftext|>',
                                              eos_token='<|endoftext|>', pad_token='<|pad|>')
    model = GPTNeoForCausalLM.from_pretrained(
        model_name).cuda() if torch.cuda.is_available() else GPTNeoForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    print("Loading dataset...")
    train_dataset, test_dataset = load_xiaohongshu_dataset(tokenizer, trial_no)

    print("Start training...")
    training_args = TrainingArguments(output_dir='results', num_train_epochs=2,
                                      logging_steps=10, load_best_model_at_end=True,
                                      evaluation_strategy='epoch',
                                      save_strategy="epoch", per_device_train_batch_size=2,
                                      per_device_eval_batch_size=2,
                                      warmup_steps=100, weight_decay=0.01, logging_dir='logs')

    Trainer(model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=test_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                                   'attention_mask': torch.stack([f[1] for f in data]),
                                                                   'labels': torch.stack([f[0] for f in data])}).train()

    # test
    print("Start testing...")
    # eval mode on model
    _ = model.eval()

    # compute prediction on test data
    original, predicted, all_text, predicted_text = [], [], [], []
    for keyword, article in tqdm(zip(test_dataset[0], test_dataset[1])):
        # predict sentiment on test data
        prompt = f'<|startoftext|>Keywords: {keyword}\nArticle:'
        generated = tokenizer(f"{prompt}", return_tensors="pt").input_ids.cuda()
        sample_outputs = model.generate(generated, do_sample=False, top_k=50, max_length=512, top_p=0.90,
                                        temperature=0, num_return_sequences=0)
        pred_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
        # extract the predicted sentiment
        try:
            pred_article = re.findall("\nArticle: (.*)", pred_text)[-1]
        except:
            pred_article = "None"
        original.append(keyword)
        predicted.append(pred_article)
        all_text.append(article)
        predicted_text.append(pred_text)
    # transform into dataframe
    df = pd.DataFrame(
        {'text': all_text, 'predicted': predicted, 'original': original, 'predicted_text': predicted_text})
    df.to_csv(f"result_run_{trial_no}.csv", index=False)
    # compute f1 score
    print(f1_score(original, predicted, average='macro'))
