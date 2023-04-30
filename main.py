from datetime import datetime
from pathlib import Path
import logging

from scrapy_spiders.habrparser import run_habrparser
from tokenizer import run_tokenizer
from ml_utils.w2v_trainer import run_training
from ml_utils.articles_util import get_tag_collection
from ml_utils.articles_util import save_tag_collection
from ml_utils.reporter import gen_report

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

if __name__ == '__main__':
  artifact_version = f'{datetime.now().timestamp()}'
  artifacts_dir = Path('artifacts') / artifact_version
  if not artifacts_dir.exists():
    artifacts_dir.mkdir(parents=True)

  model_artifact_path = str(artifacts_dir / f'w2v_model_{artifact_version}.model')
  tag_collection_artifact_path = str(artifacts_dir / f'tags_{artifact_version}')
  report_artifact_path = str(artifacts_dir / f'report_{artifact_version}.md')

  logging.info(f'Running pipeline | v {artifact_version}')
  logging.debug(f'model_artifact_path {model_artifact_path}')
  logging.debug(f'tag_collection_artifact_path {tag_collection_artifact_path}')
  logging.debug(f'report_artifact_path {report_artifact_path}')

  logging.info('STAGE 1. Collecting articles')
  run_habrparser()

  logging.info('STAGE 2. Tokenizing articles content')
  run_tokenizer()

  logging.info('STAGE 3. Train word2vec model')
  run_training(model_artifact_path)

  logging.info('STAGE 4. Prepare tag list')
  tag_collection = get_tag_collection()
  save_tag_collection(tag_collection_artifact_path, tag_collection)

  logging.info('STAGE 5. Building report')
  gen_report(model_artifact_path, report_artifact_path)

  logging.info('Done!')
