from pathlib import Path
from collections import Counter
from os import environ

import ujson
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')


# articles_dir_path = Path(environ['ARTICLES_PATH']) / 'tokenized'
articles_dir_path = Path(environ['ARTICLES_PATH'])


def get_articles_content():
    for file in articles_dir_path.iterdir():
        if file.is_file():
            with open(file, 'r', encoding='UTF-8') as f:
                article_json = ujson.loads(f.read())
                yield article_json['content']


def get_articles_json():
    for file in articles_dir_path.iterdir():
        if file.is_file():
            with open(file, 'r', encoding='UTF-8') as f:
                article_json = ujson.loads(f.read())
                yield article_json


def get_articles_tags():
    for file in articles_dir_path.iterdir():
        if file.is_file():
            with open(file, 'r', encoding='UTF-8') as f:
                article_json = ujson.loads(f.read())
                # yield list(map(lambda t: t.lower(), article_json['tags']))
                for tag in article_json['tags']:
                    yield tag


def get_tag_collection() -> set[str]:
    conter = Counter(get_articles_tags())
    tag_collection = {k: v for k, v in filter(lambda t: t[1]>=2, conter.items())}
    tag_collection = set(tag_collection)
    return tag_collection


def save_tag_collection(tag_collection_path: str, tag_collection: set[str]):
    tag_collection = list(tag_collection)
    with open(tag_collection_path, 'w', encoding="UTF-8") as f:
        result = ujson.dumps(
            tag_collection,
            sort_keys=False,
            indent=4,
            ensure_ascii=False,
            )
        f.write(result)


def get_prefiltered_data():
    top_10_tags = {
        'проблема', 'python', 'apple', 'искусственный интеллект',
        'программирование', 'игры', 'россия', 'информационная безопасность',
        'машинное обучение', 'санкции',
    }

    for article_json in get_articles_json():
        # if set(article_json['tags']) & top_10_tags:
        for tag in top_10_tags:
            for article_tag in article_json['tags']:
                if tag == article_tag:
                    yield (article_json['content'], get_tag_identificator(tag))


def get_tag_identificator(tag: str) -> int:
    return {
        'проблема': 0,
        'python': 1,
        'apple': 2,
        'искусственный интеллект': 3,
        'программирование': 4,
        'игры': 5,
        'россия': 6,
        'информационная безопасность': 7,
        'машинное обучение': 8,
        'санкции': 9,
    }[tag]



def normalize_for_bard(text: str) -> str:
    words = text.split()
    # Фильтрация стоп-слов
    stop_words = set(stopwords.words("russian"))
    filtered_words = [word for word in words if word.lower() not in stop_words]  
    # Преобразование отфильтрованных слов обратно в текст
    filtered_text = ' '.join(filtered_words)  
    
    return filtered_text