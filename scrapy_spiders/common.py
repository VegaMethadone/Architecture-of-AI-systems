from datetime import datetime
from pathlib import Path
from os import environ

import scrapy
import ujson


class HabrSpider(scrapy.Spider):
  """Generic class for HabrSpider's"""

  name = 'habrspider'


# articles_dir_path = Path('parsed_articles') / f'articles_{datetime.now().strftime("%d_%m_%Y")}'
articles_dir_path = Path(environ['ARTICLES_PATH'])
if not articles_dir_path.exists():
    articles_dir_path.mkdir(parents=True)


def save_article(article_id: str, article_name: str, content: str, tags: list[str]):
    result = ujson.dumps({
        "article_id": article_id,
        "article_name": article_name,
        "content": content,
        "tags": tags,
      },
      sort_keys=False,
      indent=4,
      ensure_ascii=False,
    )

    article_path = articles_dir_path / f'article_{article_id}.json'
    with open(str(article_path), 'w') as f:
            f.write(result)