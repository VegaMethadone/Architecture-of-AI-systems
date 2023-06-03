import os
import json
# import torch
# from transformers import BartTokenizer
# from transformers import BartForSequenceClassification
# from torch.utils.data import DataLoader, TensorDataset
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

# bard version
# model = BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels=10)
# model.load_state_dict(torch.load('service_model.pt', map_location=torch.device('cpu')))
# # load_state_dict(torch.load('classifier.pt', map_location=torch.device('cpu')))
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# linear regression version
with open('service_model.pt', 'rb') as file:
    model = pickle.load(file)
with open('vectorizer', 'rb') as file:
    vectorizer = pickle.load(file)

def get_tag_by_id(tag: str) -> int:
  return [
    'проблема',
    'python',
    'apple',
    'искусственный интеллект',
    'программирование',
    'игры',
    'россия',
    'информационная безопасность',
    'машинное обучение',
    'санкции',
  ][tag]

def normalize_text(text: str) -> str:
    normalized_content = text.lower()
    # Очистка текста от всех ненужных символов
    reg = re.compile('[^а-яА-ЯёЁa-zA-Z ]')
    normalized_content = reg.sub('', normalized_content)

    return normalized_content

def get_text_lable_old(text: str):
  text = normalize_text(text)
  print('got text endings with', text[-20:])
  encoded_inputs = tokenizer([text], padding=True, truncation=True, return_tensors='pt')
  input_ids = encoded_inputs['input_ids']
  attention_mask = encoded_inputs['attention_mask']

  with torch.no_grad():
    model.eval()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

  predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()

  print('predicted labels', predicted_labels)
  return get_tag_by_id(predicted_labels[0])

def get_text_lable(text: str):
  text = normalize_text(text)
  print('got text endings with', text[-20:])

  # vectorizer = TfidfVectorizer()
  X = vectorizer.transform([text])

  predicted_labels = model.predict(X)

  # for predicted_label in predicted_labels:
  #   print(f'Предсказанная метка: {get_tag_by_id(predicted_label)}')
  print('predicted labels', predicted_labels)
  return get_tag_by_id(predicted_labels[0])

if __name__ == '__main__':
  import sys
  text = sys.argv[1]

  lable = get_text_lable(text)
  print(lable)