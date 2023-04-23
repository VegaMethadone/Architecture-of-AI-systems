from pymystem3 import Mystem
import nltk
from nltk.corpus import stopwords

import re
import json
import os
from os import listdir
from os import path

m = Mystem()
# ntlk.downlo
stopwords = set(stopwords.words('russian'))

def prepare_text(text):
    prepared_text = text
    prepared_text = prepared_text.replace("\n", " ")
    prepared_text = re.sub(r'[^\w ]', '', prepared_text)
    prepared_text = prepared_text.lower()
    prepared_text = " ".join(filter(lambda x: x not in stopwords, prepared_text.split(" ")))
    prepared_text = ''.join(m.lemmatize(prepared_text))
    prepared_text = prepared_text.strip()
    prepared_text = prepared_text[:8000]
    
    return prepared_text

def prepare_tags(tags):
    prepared_tags = tags
    prepared_tags = map(lambda tag: tag.lower(), prepared_tags)
    prepared_tags = map(lambda tag: tag.replace("-", " "), prepared_tags)
    
    return list(prepared_tags)



articles_dir = "../parsed_articles/1000_articles_11_04_2023"
raw_articles_dir = path.join(articles_dir, "raw")
normalized_articles_dir = path.join(articles_dir, "normalized")

if not os.path.exists(normalized_articles_dir):
    os.makedirs(normalized_articles_dir)

article_files = [f for f in listdir(raw_articles_dir) if path.isfile(path.join(raw_articles_dir, f))]

#articles = dict()

for article_file in article_files:
    article_path = path.join(raw_articles_dir, article_file)
    # print(article_path)
    print(f"Processing {article_path} ...")
    with open(article_path, 'r', encoding="UTF-8") as f:
        article = f.read()
        article_json = json.loads(article)
        #articles[article_json["article_id"]] = article_json
        # print(article_json)
        article_json['content'] = prepare_text(article_json['content'])
        article_json['tags'] = prepare_tags(article_json['tags'])

        result = json.dumps({**article_json},
        sort_keys=False,
        indent=4,
        ensure_ascii=False,
        )
         
        with open(path.join(normalized_articles_dir, article_file), 'w', encoding="UTF-8") as f:
            f.write(result)
