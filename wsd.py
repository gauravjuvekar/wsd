#!/usr/bin/env python
import SIF
import sent2vec

import scipy
import scipy.spatial
import scipy.spatial.distance

import nltk

wordnet = nltk.wordnet.wordnet
wordnet.ensure_loaded()


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


def detok_sent(sent):
    detokenizer = MosesDetokenizer()
    return detokenizer.detokenize(sent, return_str=True)


def s2v_embeds(sent_list):
    return [s2v_model.embed_sentence(detok_sent(sent)) for sent in sent_list]


def get_hypernyms(word):
    hyp_dict = {}
    for syn in wordnet.synsets(word):
        #hypernyms() returns a list of hypernyms
        hyp_dict[syn] = syn.hypernyms()
    #dictionary with each sense, and corresponding hypernym
    return hyp_dict


def replace_target_word(tok_sent, index):
    # given a sentence represented as a list of tokens, and the index of the
    # token to be replaced (i.e the token to be disambiguated), return list of
    # sentences with hypernym replaced for each sense of target token word
    hyp_dict = get_hypernyms(tok_sent[index])
    lemset = set()
    for elem in hyp_dict:
        if len(hyp_dict[elem]) == 0: #if no hypernym, replace with the synonym
            for lem in elem.lemmas():
                lemset.add(str(lem.name()))
        for hyp in hyp_dict[elem]:
            for lem in hyp.lemmas():
                lemset.add(str(lem.name()))

    sent_list = []
    for elem in sorted(lemset):
        elem_list = elem.split("_")
        new_sent = tok_sent[:index] + elem_list + tok_sent[index + 1:]
        sent_list.append(new_sent)
    return sent_list


if __name__ == '__main__':
    import pprint
    if True:

        print("sentences")
        pprint.pprint(sentences)
        print("SIF embeddings")
        pprint.pprint(sif_embeds(sentences))
        print("S2V embeddings")
        pprint.pprint(s2v_embeds(sentences))

    if True:
        sentences = ["This is a sentence".split(),
                     "And this is another one".split()]
        index = (1, 3)

        sent_list = replace_target_word(sentences[index[0]], index[1])
        orig_sent = sentences[index[0]]

