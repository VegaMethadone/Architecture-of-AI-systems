import logging

import gensim

from ml_utils.articles_util import get_articles_content


def run_training(artifact_name: str):
    logging.info('Reading documents..')
    documents = tuple(get_articles_content())
    logging.info(f'Got {len(documents)} documents')

    logging.info('Run model training...')
    model = gensim.models.Word2Vec(sentences=documents, vector_size=100, window=20, min_count=2, workers=10)
    model.train(documents, total_examples=len(documents), epochs=10)
    # logging.info('Model trained')

    # artifact_name = f'w2v_model_{datetime.now().timestamp()}.model' if not artifact_name else artifact_name
    model.save(artifact_name)
    logging.info(f'Model saved: {artifact_name}')


if __name__ == "__main__":
    run_training()