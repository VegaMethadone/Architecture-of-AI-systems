from scrapy_spiders.habrparser import run_habrparser
from tokenizer import run_tokenizer

if __name__ == "__main__":
  print("STAGE 1. Collecting articles")
  # run_habrparser()
  print("STAGE 2. Tokenizing articles content")
  run_tokenizer()
  print("STAGE 3. Train word2vec model")
