import logging

# import gensim
import os
import json
import torch
from tqdm import tqdm
from transformers import BartTokenizer
from transformers import BartForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

from ml_utils.articles_util import get_articles_content, get_prefiltered_data, normalize_for_bard


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

def run_svm_training(artifact_name: str):
    logging.info('Reading documents..')
    documents = tuple(get_articles_content())
    logging.info(f'Got {len(documents)} documents')

    logging.info('Run model training...')
    # model = gensim.models.Word2Vec(sentences=documents, vector_size=100, window=20, min_count=2, workers=10)
    # model.train(documents, total_examples=len(documents), epochs=10)
    # logging.info('Model trained')


    # artifact_name = f'w2v_model_{datetime.now().timestamp()}.model' if not artifact_name else artifact_name
    model.save(artifact_name)
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


if __name__ == "__main__":
    import sys

    model_type = sys.argv[1]

    train_func = {
      'bard': run_bard_training,
      'svm': run_svm_training,
    }[model_type]

    print(model_type, train_func)
    # run_bard_training("test_model.pt")
    train_func("rest_service/service_model.pt")