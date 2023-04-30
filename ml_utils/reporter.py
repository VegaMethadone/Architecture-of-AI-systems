import sys

import gensim

from ml_utils.articles_util import get_articles_json
from ml_utils.articles_util import get_tag_collection
from ml_utils.predictions import lable_text
from ml_utils.predictions import lable_text_v2


def gen_report(model_path: str, report_path: str):
    model = gensim.models.Word2Vec.load(model_path)
    
    articles_json_gen = get_articles_json()
    tag_collection = get_tag_collection()

    with open(report_path, 'w') as report:
        report.write('## Report v1\n')

        for i in range(50):
            article_json = next(articles_json_gen)
            article_name = article_json['article_name']
            article_content = article_json['content']

            tags = lable_text(article_content, model, tag_collection, 10)
            # tags = lable_text_v2(article_content, model, tag_collection, 10)

            
            report.write(f'### {article_name}\n')
            report.write(f'{" ".join(article_content[:50])}\n')
            for tag in tags:
                report.write(f'- **{tag[0]}** - {tag[1]}\n')
            report.write('\n')
            report.write('')


if __name__ == "__main__":
    model_path = sys.argv[1]

    gen_report(model_path, 'report.md')