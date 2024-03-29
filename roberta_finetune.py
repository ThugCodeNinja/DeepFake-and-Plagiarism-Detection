import os
import numpy as np 
import pandas as pd 
from tqdm import tqdm
gpt_path='/kaggle/input/chatgpt/full_texts/chatgpt'
human_path='/kaggle/input/chatgpt/full_texts/human'
import os
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def preprocess_text(text):
    # Remove special characters and extra spaces
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Tokenize the cleaned text
    tokenized_text = tokenizer(cleaned_text, truncation=True, padding=True)
    
    return tokenized_text

X = []
Y = []

# Process each file in the directory
for file_name in tqdm(os.listdir(gpt_path)):
    if file_name.endswith('.txt'):
        file_path = os.path.join(gpt_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Preprocess the text
        tokenized_text = preprocess_text(text)
        
        # Append tokenized text to X
        X.append(tokenized_text['input_ids'])
        
        Y.append(1)
for file_name in tqdm(os.listdir(human_path)):
    if file_name.endswith('.txt'):
        file_path = os.path.join(human_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Preprocess the text
        tokenized_text = preprocess_text(text)
        
        # Append tokenized text to X
        X.append(tokenized_text['input_ids'])      
        Y.append(0)

# Pad sequences to the same length
max_length = max(len(ids) for ids in X)
X = [ids + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in X]

# Convert X and Y to torch tensors
input_ids = torch.tensor(X)
attention_mask = torch.tensor([[1] * len(ids) + [0] * (max_length - len(ids)) for ids in X])
labels = torch.tensor(Y)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
    output_dir='/kaggle/working/'
)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Initialize RoBERTa model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=Dataset({'input_ids': X_train, 'attention_mask': attention_mask[:len(X_train)]}, labels= y_train),
    eval_dataset=Dataset({'input_ids': X_val, 'attention_mask': attention_mask[len(X_train):len(X_train)+len(X_val)]}, labels=y_val),
)
# Fine-tune the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)
# Define the directory where you want to save the model
output_dir = "/kaggle/working/finetune"

# Save the model
model.save_pretrained(output_dir)

# Save the tokenizer as well, in case you need it for inference
tokenizer.save_pretrained(output_dir)

