from scrapy_spiders.common import HabrSpider
from scrapy_spiders.common import save_article

# dir_path = f'parsed_articles/articles_{datetime.now().strftime("%d_%m_%Y")}'
# if not os.path.exists(dir_path):
#     os.makedirs(dir_path)

class HabrSpiderLarge(HabrSpider):
  start_urls = (f"https://habr.com/ru/articles/{article_id}" for article_id in range(729084, 600000, -2))

  def parse(self, response):
    content = "\n".join(response.css("#post-content-body *::text").getall())
    tags = response.css(".tm-tags-list__link::text").getall()
    article_name = response.css(".tm-title_h1 span ::text").get()
    article_id = response.url.split("/")[-2]

    save_article(article_id, article_name, content, tags)
