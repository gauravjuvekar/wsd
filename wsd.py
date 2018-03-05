#!/usr/bin/env python

import SIF
import sent2vec

import scipy
import scipy.spatial
import scipy.spatial.distance

ENABLE = ('sif', 's2v')

sif_db = SIF.data_io.setup_db('./data/sif.db')
s2v_model = sent2vec.Sent2vecModel()

if 's2v' in ENABLE:
    s2v_model.load_model('./data/s2v_wiki_unigrams.bin')

def sif_embeds(sent_list):
    idx_mat, weight_mat, data = SIF.data_io.prepare_data(sent_list, sif_db)
    params = SIF.params.params()
    params.rmpc = 1
    embedding = SIF.SIF_embedding.SIF_embedding(idx_mat,
                                                weight_mat,
                                                data,
                                                params)
    return list(embedding)

def s2v_embeds(sent_list):
    return [s2v_model.embed_sentence(' '.join(sent)) for sent in sent_list]


if __name__ == '__main__':
    sentences = ["this is a sentence".split(),
                 "and this is another one".split()]

    import pprint
    print(sentences)
    pprint.pprint(sentences)
    print("SIF embeddings")
    pprint.pprint(sif_embeds(sentences))
    print("S2V embeddings")
    pprint.pprint(s2v_embeds(sentences))



