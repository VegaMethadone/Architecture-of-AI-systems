from pathlib import Path
from collections import Counter
from os import environ

import ujson


articles_dir_path = Path(environ['ARTICLES_PATH']) / 'tokenized'


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


