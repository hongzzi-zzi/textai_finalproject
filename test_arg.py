import torch
import pandas as pd
import os
import argparse
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import Trainer
from transformers import BertForTokenClassification, BertTokenizer, ElectraTokenizer, ElectraForTokenClassification, DistilBertForTokenClassification, DistilBertTokenizer, DistilBertConfig
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
#%%
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
# https://chat.openai.com/share/45e6d0f5-60e4-4cd4-8063-6e0ddaf51591
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='koelectra-small', choices=['koelectra-small', 'distil_kobert', 'kobert'],help="Model name")
    parser.add_argument("--test_dataset_path", type=str, default='/home/hongeun/23-2/text_ai/final/textai_finalproject/test.csv', help="Path to the test data file")
    parser.add_argument("--label_path", type=str, default='/home/hongeun/23-2/text_ai/final/textai_finalproject/PII_label.txt', help="Logging directory")
    
    args = parser.parse_args()
    
    print('--------------------------------------------------')
    print(args)
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_path='/home/hongeun/23-2/text_ai/final/textai_finalproject'
    
    ner_test_df = pd.read_csv(args.test_dataset_path, delimiter='\t', header=None, names=['Sentence #', 'Word'])
    with open(args.label_path, 'r') as f:
        labels = f.read().splitlines()
    label_map = {label: i for i, label in enumerate(labels)}
    
    test_sentences = ner_test_df['Sentence #'].tolist()
    test_ner_tags = ner_test_df['Word'].apply(lambda x: x.split(' ')).tolist()


    # 모델 로드
    if 'koelectra-small' in args.model_name:
        model_path = os.path.join(root_path,'koelectra-small', 'results', 'best_model')
        # print(model_path)
        model = ElectraForTokenClassification.from_pretrained(model_path).to(device)
        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
    else:
        raise ValueError("The model path doesn't match any known models")
    
    
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

    max_len = 128
    test_dataset = NERDataset(test_sentences, test_ner_tags, label_map, tokenizer, max_len)
    trainer = Trainer(model=model,eval_dataset=test_dataset,compute_metrics=compute_metrics)
    results = trainer.evaluate(test_dataset)
    print(results)


if __name__ == "__main__":
    main()
