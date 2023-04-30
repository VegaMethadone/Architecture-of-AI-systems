import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy_spiders.habrparser_fast import HabrSpiderFast


def run_habrparser():
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'
    })

    process.crawl(HabrSpiderFast)
    process.start() # the script will block here until the crawling is finished


if __name__ == "__main__":
    run_habrparser()
