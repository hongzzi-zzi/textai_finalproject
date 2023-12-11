#%%
## import lib
import torch
import pandas as pd
import os
import argparse
import numpy as np
from copy import deepcopy
from transformers import TrainerCallback

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import BertForTokenClassification, BertTokenizer, ElectraTokenizer, ElectraForTokenClassification, DistilBertForTokenClassification, DistilBertTokenizer
#%%

class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

## custom dataset for NER
class NERDataset(Dataset):
    def __init__(self, sentences, labels, label_map, tokenizer, max_len=128):
        self.sentences = sentences  # list of sentences
        self.labels = labels  # list of label sequences for each sentence
        self.label_map = label_map  # list of label sequences for each sentence
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        # labels=[self.label_map.get(l, self.label_map['UNK']) for l in self.labels[idx]]
        labels=[self.label_map.get(l) for l in self.labels[idx]]
        
        # Encode the sentence and labels
        encoding = self.tokenizer(
            sentence,
            add_special_tokens=False,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )

        labels = labels + [-100] * (self.max_len - len(labels))
        labels = labels[:self.max_len]

        input_ids=encoding['input_ids'].squeeze()
        attention_mask=encoding['attention_mask'].squeeze()
        labels=torch.tensor(labels, dtype=torch.long)

        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'labels': labels
        }

# %%

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='koelectra-small', choices=['koelectra-small', 'distil_kobert', 'kobert'],help="Model name")
    parser.add_argument("--train_dataset_path", type=str, default='/home/hongeun/23-2/text_ai/final/textai_finalproject/train.csv', help="Logging directory")
    parser.add_argument("--label_path", type=str, default='/home/hongeun/23-2/text_ai/final/textai_finalproject/PII_label.txt', help="Logging directory")
    parser.add_argument("--logging_and_output_root_dir", type=str, default='/home/hongeun/23-2/text_ai/final/textai_finalproject/', help="Logging directory")
    parser.add_argument("--logging_dir", type=str, default='/home/hongeun/23-2/text_ai/final/textai_finalproject/', help="Logging directory")
    parser.add_argument("--num_train_epochs", type=int, default=15, help="Number of training epochs")

    args = parser.parse_args()
    
    print('--------------------------------------------------')
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model_name 
    train_dataset_path = args.train_dataset_path
    label_path = args.label_path
    # output_dir = args.output_dir
    # logging_dir = args.logging_dir
    root_dir=args.logging_and_output_root_dir
    num_train_epochs = args.num_train_epochs
    
    # data load
    ner_train_df = pd.read_csv(train_dataset_path, delimiter='\t', header=None, names=['Sentence #', 'Word'])
    with open(label_path, 'r') as f:
        labels = f.read().splitlines()
    label_map = {label: i for i, label in enumerate(labels)}

    train_sentences = ner_train_df['Sentence #'].tolist()
    train_ner_tags = ner_train_df['Word'].apply(lambda x: x.split(' ')).tolist()

    train_sentences, val_sentences, train_ner_tags, val_ner_tags = train_test_split(train_sentences, train_ner_tags, test_size=0.1, random_state=42)
    
    
    if model_name=='koelectra-small':
        model = ElectraForTokenClassification.from_pretrained("monologg/koelectra-small-v3-discriminator", num_labels=len(labels)).to(device)
        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
        output_dir = os.path.join(root_dir, 'koelectra-small/results')
        logging_dir = os.path.join(root_dir, 'koelectra-small/logs')

    # elif model_name=='distil_kobert':
    #     model = DistilBertForTokenClassification.from_pretrained("monologg/distilkobert", num_labels=len(labels)).to(device)
    #     tokenizer = DistilBertTokenizer.from_pretrained("monologg/distilkobert")
    #     output_dir = os.path.join(root_dir, 'distil_kobert/results')
    #     logging_dir = os.path.join(root_dir, 'distil_kobert/logs')

    # elif model_name=='kobert':
    #     model = BertForTokenClassification.from_pretrained("monologg/kobert", num_labels=len(labels)).to(device)
    #     tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
    #     output_dir = os.path.join(root_dir, 'kobert/results')
    #     logging_dir = os.path.join(root_dir, 'kobert/logs')
    else:
        raise ValueError("The model path doesn't match any known models")

    max_len = 128  
    train_dataset = NERDataset(train_sentences, train_ner_tags, label_map, tokenizer, max_len)
    val_dataset = NERDataset(val_sentences, val_ner_tags, label_map, tokenizer, max_len)
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels = labels.flatten()
        true_predictions = predictions.flatten()

        true_predictions = true_predictions[true_labels != -100]
        true_labels = true_labels[true_labels != -100]

        accuracy = accuracy_score(true_labels, true_predictions)
        
        return {
            'accuracy': accuracy,
        }
        

    training_args = TrainingArguments(
        output_dir=output_dir,   
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=256,  
        per_device_eval_batch_size=256,   
        warmup_steps=500,     
        weight_decay=0.01,       
        logging_dir=logging_dir,      
        logging_strategy="epoch", 
        do_eval=True,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        fp16=True,
        learning_rate=1e-5,
        log_level='info',
        seed=15,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset= train_dataset,
        eval_dataset = val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        compute_metrics=compute_metrics, 
        
    )
    
    
    trainer.add_callback(CustomCallback(trainer)) 

    train = trainer.train()
    
    # 새로운 저장 경로 설정
    best_model_checkpoint = trainer.state.best_model_checkpoint
    new_save_path = os.path.join(output_dir, 'best_model')
    trainer.save_model(new_save_path)
    print(f"Best model saved to: {new_save_path}")
    print()


if __name__ == "__main__":
    main()
# %%
