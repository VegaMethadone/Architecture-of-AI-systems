import threading

habr_link = "https://habr.com/ru/all/"


import scrapy

# ITEM_PIPELINES = {
#     'myproject.pipelines.PricePipeline': 300,
#     'myproject.pipelines.JsonWriterPipeline': 800,
# }

# class PricePipeline:
#     vat_factor = 1.15

#     def process_item(self, item, spider):
#         adapter = ItemAdapter(item)
#         if adapter.get('price'):
#             if adapter.get('price_excludes_vat'):
#                 adapter['price'] = adapter['price'] * self.vat_factor
#             return item
#         else:
#             raise DropItem(f"Missing price in {item}")

class BlogSpider(scrapy.Spider):
  name = 'habrspider'
  # start_urls = [habr_link]
  start_urls = (f"https://habr.com/ru/all/page{page}" for page in range(3))

  def parse(self, response):
    for article in response.css('.tm-title__link'):
      article_link = article.xpath('@href').get()
      yield response.follow(article_link, callback=self.parse_article)

  def parse_article(self, response):
    content = "".join(response.css("#post-content-body *::text").getall())
    tags = response.css(".tm-tags-list__link::text").getall()

    # print(content)
    # print(tags)
    yield {
      "content": content,
      "tags": tags,
    }