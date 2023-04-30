import numpy as np
# from scipy import spatial
from gensim.models import Word2Vec



# def model_similarity (model, list1, list2, l1neg=[], l2neg=[]):
#     """finds the similarity according to a model of two lists of words
#     also accepts 'negative' lists for analogies."""
#     list1sum=sum([model[l] for l in list1])
#     list2sum=sum([model[l] for l in list2])
#     if l1neg: list1sum -= sum([model[l] for l in l1neg])
#     if l2neg: list1sum -= sum([model[l] for l in l2neg])
#     return 1-spatial.distance.cosine(list1sum,list2sum)


def lable_text(text: list[str], model: Word2Vec, tag_collection: set[str], n=3, threshold=0.0):
    text_words = list(filter(lambda x: x in model.wv, text))
    text_vectors = list(map(lambda x: model.wv.get_vector(x, norm=True), text_words))
    text_avg_sum_vector = sum(text_vectors) / len(text_vectors)
    
    lables = [] # (lable, similarity)
    for tag in tag_collection:
        if tag in model.wv:
            tag_vec = model.wv.get_vector(tag, norm=True)
            # tag_vec = model.wv[tag]
            x = np.dot(text_avg_sum_vector, tag_vec)
            if x < threshold:
                continue
            if len(lables) < n:
                lables.append((tag, x))
            elif tag in map(lambda l: l[0], lables):
                continue
            else:
                least_tag = min(lables, key=lambda l: l[1])
                if x > least_tag[1]:
                    least_tag_index = lables.index(least_tag)
                    lables[least_tag_index] = (tag, x)
    # return lables
    return sorted(lables, key=lambda l: l[1], reverse=True)


def lable_text_v2(text: list[str], model: Word2Vec, n=3):
    text_words = list(filter(lambda x: x in model.wv, text))
    text_vectors = list(map(lambda x: model.wv.get_vector(x, norm=True), text_words))
    text_avg_sum_vector = sum(text_vectors) / len(text_vectors)
    
    lables = model.wv.most_similar(positive=text_avg_sum_vector, topn=n)
    return lables
