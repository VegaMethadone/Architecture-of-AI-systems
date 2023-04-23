import scrapy
import os
import json
from datetime import datetime

dir_path = f'parsed_articles/articles_{datetime.now().strftime("%d_%m_%Y")}'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

class HabrSpiderLarge(scrapy.Spider):
  name = 'habrspider'
  start_urls = (f"https://habr.com/ru/articles/{article_id}" for article_id in range(729084, 600000, -1))

  def parse(self, response):
    content = "\n".join(response.css("#post-content-body *::text").getall())
    tags = response.css(".tm-tags-list__link::text").getall()
    article_name = response.css(".tm-title_h1 span ::text").get()
    article_id = response.url.split("/")[-2]

    result = json.dumps({
        "article_id": article_id,
        "article_name": article_name,
        "content": content,
        "tags": tags,
      },
      sort_keys=False,
      indent=4,
      ensure_ascii=False,
    )

    with open(f"{dir_path}/article_{article_id}.json", 'w') as f:
            f.write(result)