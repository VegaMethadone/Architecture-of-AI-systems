# Architecture-of-AI-systems



### Demo example

run commands from project root



to collect last 1000 articles:

```bash
scrapy runspider src/scrapy_spiders/habrparser_common.py
```



normalize them:

```bash
python src/normalizer.py
```



... model training etc ...