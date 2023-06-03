import logging

# import gensim
import time
import pickle
import itertools
import os
import json
import torch
from os import environ
from pathlib import Path
from tqdm import tqdm
from transformers import BartTokenizer
from transformers import BartForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

from ml_utils.articles_util import get_articles_content, get_prefiltered_data, normalize_for_bard, count_articles, count_prefiltered_data


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

# articles_amount = count_articles()
articles_amount = count_prefiltered_data()
models_dir_path = Path(environ['MODELS_DIR_PATH'])
if not models_dir_path.exists():
        models_dir_path.mkdir(parents=True)


def get_part_of_data(a, b, iterable):
    drop_amount = int(a * articles_amount / 100)
    take_amount = int(b * articles_amount / 100)

    print(drop_amount, take_amount)

    i = 0

    def drop_func(elem):
        nonlocal i
        i += 1
        return i < drop_amount

    def take_func(elem):
        nonlocal i
        i += 1
        return i < take_amount

    return itertools.takewhile(take_func , itertools.dropwhile(drop_func, iterable))


def run_linear_model_training(artifact_name: str):
    logging.info('Run model training...')
    # model = gensim.models.Word2Vec(sentences=documents, vector_size=100, window=20, min_count=2, workers=10)
    # model.train(documents, total_examples=len(documents), epochs=10)
    # logging.info('Model trained')
    texts = []
    labels = []

    for text, tag_id in tqdm(get_part_of_data(0, 100, get_prefiltered_data())):
    # for text, tag_id in tqdm(get_prefiltered_data()):
        text = normalize_for_bard(text)

        texts.append(text)
        labels.append(tag_id)


    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)
    # print(classification_report(y_test, y_pred))
    # print("Accuracy:", accuracy)
    # artifact_name = f'w2v_model_{datetime.now().timestamp()}.model' if not artifact_name else artifact_name
    import pickle
    with open(f'{artifact_name}', 'wb') as file:
        pickle.dump(model, file)
    # model.save(artifact_name)
    with open('vectorizer', 'wb') as file:
        pickle.dump(vectorizer, file)
        
    logging.info(f'Model saved: {artifact_name}')


def run_bard_training(artifact_name: str):
    # logging.info('Reading documents..')
    logging.info('Run model training...')
    # model = gensim.models.Word2Vec(sentences=documents, vector_size=100, window=20, min_count=2, workers=10)
    # model.train(documents, total_examples=len(documents), epochs=10)
    # logging.info('Model trained')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')



    max_length = 1000

    # Преобразование текстов в последовательности токенов и паддинг
    input_ids = []
    attention_masks = []

    y_train = []  # labels

    for text, tag_id in tqdm(get_prefiltered_data()):
        text = normalize_for_bard(text)

        encoded = tokenizer.encode_plus(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'].squeeze())
        attention_masks.append(encoded['attention_mask'].squeeze())
        y_train.append(tag_id)

    # Преобразование в тензоры PyTorch
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)

    # Вывод размерности тензоров
    logging.info(f"Input IDs shape: {input_ids.shape}")
    logging.info(f"Attention Masks shape: {attention_masks.shape}")


    # encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    # input_ids = encoded_inputs['input_ids']
    # attention_mask = encoded_inputs['attention_mask']
    labels = y_train
    dataset = TensorDataset(input_ids, attention_masks, torch.tensor(labels))



    model = BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    model.to(device)
    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in tqdm(train_dataloader):
            batch = [item.to(device) for item in batch]
            input_ids, attention_mask, labels = batch
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {average_loss:.4f}")
    
    # Сохранение модели:
    try:
        torch.save(model.state_dict(), artifact_name)
        logging.info(f'Model saved: {artifact_name}')
    except SystemError:
        logging.error(f'Model saving error')


# def additional_train_linear_model(old_model_name: str, new_model_name: str):
def additional_train_linear_model(model_name: str):
    old_model_name = model_name
    new_model_name = model_name

    with open(old_model_name, 'rb') as file:
        loaded_model = pickle.load(file)

    texts = []
    labels = []

    for text, tag_id in tqdm(get_part_of_data(0, 10, get_prefiltered_data())):
        text = normalize_for_bard(text)
        texts.append(text)
        labels.append(tag_id)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    new_X = vectorizer.fit_transform(texts)
    new_y = np.array(labels)

    loaded_model.fit(new_X, new_y)

    with open(new_model_name, 'wb') as file:
        pickle.dump(loaded_model, file)

    logging.info(f'Model saved: {model_name}')



def measure_linear_model() -> float:
    texts = []
    labels = []

    for text, tag_id in tqdm(get_part_of_data(20, 100, get_prefiltered_data())):
        text = normalize_for_bard(text)
        texts.append(text)
        labels.append(tag_id)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    from sklearn.metrics import accuracy_score

    time_start = time.time()
    try:

        model_path = models_dir_path / 'service_model.pt'
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)

        
        y_pred = loaded_model.predict(X)
        accuracy = accuracy_score(y, y_pred)
    except:
        print("Warning: model tests failed")

    time_end = time.time()

    return time_end - time_start


def measure_bard_model() -> float:
    return 3600 * 5;


if __name__ == "__main__":
    import sys

    model_type = sys.argv[1]

    train_func = {
      'bard': run_bard_training,
      # 'svm': run_svm_training,
      'linear': run_linear_model_training,
      'add_train_linear': additional_train_linear_model,
    }[model_type]

    print(model_type, train_func)
    # run_bard_training("test_model.pt")
    model_path = models_dir_path / 'service_model.pt'

    train_func(str(model_path))