from pymystem3 import Mystem
from nltk.corpus import stopwords
from tqdm import tqdm
import nltk
import ujson

import re
from os import environ
from pathlib import Path


articles_dir_path = Path(environ['ARTICLES_PATH'])


m = Mystem()
nltk.download('stopwords')
stopwords = set(stopwords.words('russian')) | set(stopwords.words('english'))

regexp = re.compile(r'[^a-zа-я0-9 ]+')

def prepare_text(text) -> list[str]:
    # prepared_text = text
    prepared_text = text.lower()
    prepared_text = prepared_text.replace('ё', 'е')
    prepared_text = re.sub(r'[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u3000\uFEFF\s]+', ' ', prepared_text)  # remove all strange spaces (such as no-break-space)
    prepared_text = re.sub(regexp, '', prepared_text)
    lemmatized = m.lemmatize(prepared_text)
    # prepared_text = " ".join(filter(lambda x: x not in stopwords and len(x) < 30, prepared_text.split(" ")))
    return list(filter(lambda x: x.strip() and  len(x) < 30 and not x.isdigit() and x not in stopwords, lemmatized))
    # return list(filter(lambda x: x.strip(), m.lemmatize(prepared_text)))
    # # prepared_text = ''.join(m.lemmatize(prepared_text))
    # # prepared_text = prepared_text.strip()
    # # return prepared_text

def prepare_tags(tags) -> list[str]:
    prepared_tags = tags
    prepared_tags = map(lambda tag: tag.lower(), prepared_tags)
    # prepared_tags = map(lambda tag: tag.replace("-", ""), prepared_tags)
    prepared_tags = map(lambda tag: re.sub(regexp, '', tag), prepared_tags)
    
    return list(prepared_tags)


def run_tokenizer():
    processed_articles_dir = articles_dir_path / 'tokenized'

    if not processed_articles_dir.exists():
        processed_articles_dir.mkdir(parents=True)

    article_files = [f for f in articles_dir_path.iterdir() if (articles_dir_path / f).is_file()]

    print(f'Running tokiniezer...')
    print(f'Processing {len(article_files)} documents')
    for article_file in tqdm(article_files):
        article_path = articles_dir_path / article_file
        with open(article_path, 'r', encoding="UTF-8") as f:
            article = f.read()
            article_json = ujson.loads(article)
            article_json['content'] = prepare_text(article_json['content'])
            article_json['tags'] = prepare_tags(article_json['tags'])

            result = ujson.dumps(
                {**article_json},
                sort_keys=False,
                indent=4,
                ensure_ascii=False,
                )

            tokinized_data_file = processed_articles_dir / article_file.name
            with open(str(tokinized_data_file), 'w', encoding="UTF-8") as f:
                f.write(result)
    print(f"Done tokenization")

if __name__ == "__main__":
    run_tokenizer()