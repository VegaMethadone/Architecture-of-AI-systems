from scrapy_spiders.common import HabrSpider
from scrapy_spiders.common import save_article

# dir_path = f'parsed_articles/articles_{datetime.now().strftime("%d_%m_%Y")}'
# if not os.path.exists(dir_path):
#     os.makedirs(dir_path)

class HabrSpiderFast(HabrSpider):
  # crawling only last 1000 articles
  start_urls = (f"https://habr.com/ru/all/page{page}" for page in range(1,51))

  def parse(self, response):
    for article in response.css('.tm-title__link'):
      article_link = article.xpath('@href').get()
      yield response.follow(article_link, callback=self.parse_article)

  def parse_article(self, response):
    content = "\n".join(response.css("#post-content-body *::text").getall())
    tags = response.css(".tm-tags-list__link::text").getall()
    article_name = response.css(".tm-title_h1 span ::text").get()
    article_id = response.url.split("/")[-2]

    save_article(article_id, article_name, content, tags)
