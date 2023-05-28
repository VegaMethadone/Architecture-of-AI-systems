# from pymystem3 import Mystem
# from nltk.corpus import stopwords
from tqdm import tqdm
# import nltk
import ujson

import re
from os import environ
from pathlib import Path


articles_dir_path = Path(environ['RAW_ARTICLES_PATH'])
normalized_articles_dir_path = Path(environ['ARTICLES_PATH'])

def normalize_text(text: str) -> str:
    normalized_content = text.lower()
    # Очистка текста от всех ненужных символов
    reg = re.compile('[^а-яА-ЯёЁa-zA-Z ]')
    normalized_content = reg.sub('', normalized_content)

    return normalized_content

def run_normalizer():
    if not normalized_articles_dir_path.exists():
        normalized_articles_dir_path.mkdir(parents=True)

    article_files = [f for f in articles_dir_path.iterdir() if (articles_dir_path / f).is_file()]

    print(f'Running normalizer...')
    print(f'Processing {len(article_files)} documents')
    for article_file in tqdm(article_files):
        article_path = articles_dir_path / article_file
        with open(article_path, 'r', encoding="UTF-8") as f:
            article = f.read()
            article_json = ujson.loads(article)
            article_json['content'] = normalize_text(article_json['content'])

            result = ujson.dumps(
                {**article_json},
                sort_keys=False,
                indent=4,
                ensure_ascii=False,
                )

            normalized_data_file = normalized_articles_dir_path / article_file.name
            with open(str(normalized_data_file), 'w', encoding="UTF-8") as f:
                f.write(result)
    print(f"Done normalization")

if __name__ == "__main__":
    run_normalizer()