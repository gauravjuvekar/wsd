#!/usr/bin/env python
import SIF
import sent2vec
import semcor_reader
import scipy
import scipy.spatial
import scipy.spatial.distance

import nltk
import nltk.tokenize
import nltk.tokenize.moses

import functools

wordnet = nltk.wordnet.wordnet
wordnet.ensure_loaded()


ENABLE = (
    'sif',
    # 's2v',
    )

sif_db = SIF.data_io.setup_db('./data/sif.db')
s2v_model = sent2vec.Sent2vecModel()

if 's2v' in ENABLE:
    s2v_model.load_model('./data/s2v_wiki_unigrams.bin')


def sif_embeds_wrapper(sent_list):
    idx_mat, weight_mat, data = SIF.data_io.prepare_data(sent_list, sif_db)
    params = SIF.params.params()
    params.rmpc = 1
    embedding = SIF.SIF_embedding.SIF_embedding(idx_mat,
                                                weight_mat,
                                                data,
                                                params)
    return list(embedding)


# def detok_sent(sent):
    # detokenizer = nltk.tokenize.moses.MosesDetokenizer()
    # return detokenizer.detokenize(sent, return_str=True)

@functools.lru_cache()
def detok_sent(sent):
    return ' '.join(sent)


@functools.lru_cache()
def s2v_embed_wrapper(sent):
    return s2v_model.embed_sentence(sent)


def s2v_embeds(sent_list):
    return [s2v_embed_wrapper(detok_sent(sent)) for sent in sent_list]


@functools.lru_cache()
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
    for synset, hypernyms in hyp_dict.items():
        if not len(hypernyms):
            for lem in synset.lemmas():
                lemset.add((lem, synset))
        for hyp in hypernyms:
            for lem in hyp.lemmas():
                lemset.add((lem, synset))

    sent_list = []
    for hyp_lemma, synset in sorted(lemset, key=lambda x: x[0]):
        elem_list = [x for y in lemma.name().split("_") for x in y.split()]
        new_sent = tok_sent[:index] + elem_list + tok_sent[index + 1:]
        sent_list.append((new_sent, synset))
    return sent_list


def choose_sense(sentences, index_to_replace, replacements,
                 embed_func, distance_func):
    # initial_embedding = embed_func(sentences)
    # orig_sentence = sentences[s_idx]
    # orig_embed = initial_embedding[s_idx]

    average_dist = []
    for replacement in replacements:
        sentences[s_idx] = replacement
        embed = embed_func(sentences)
        new_embed = embed[s_idx]
        pairwise_dist = [distance_func(new_embed, context_embed)
                         for context_embed in new_embed]
        # distance with itself will be 0
        average_dist.append((sum(pairwise_dist) /
                            (len(pairwise_dist) - 1)))

    if len(average_dist):
        min_avg_dist_i = average_dist.index(min(average_dist))
        return min_avg_dist_i



if __name__ == '__main__':
    import pprint
    if False:
        sentences = ["This is a sentence".split(),
                     "And this is another one".split()]
        print("sentences")
        pprint.pprint(sentences)
        print("SIF embeddings")
        pprint.pprint(sif_embeds(sentences))
        print("S2V embeddings")
        pprint.pprint(s2v_embeds(sentences))

    if False:
        sentences = ["This is a sentence".split(),
                     "And this is another one".split()]
        index = (1, 3)

        sent_list = replace_target_word(sentences[index[0]], index[1])
        orig_sent = sentences[index[0]]

    if True:
        semcor_file = './data/datasets/semcor3.0/brownv/tagfiles/br-r01'
        with open(semcor_file, 'rb') as f:
            paras = semcor_reader.readsemcor(f)

        count_correct = 0
        count_wrong = 0
        count_skipped = 0
        for para in paras:
            sentences = []
            indices = []
            for s_idx, sentence in enumerate(para):
                sent = []
                for w_idx, word_tup in enumerate(sentence):
                    word, lemma = word_tup
                    if lemma is None:
                        pass
                    elif isinstance(lemma, str):
                        print("No lemma for word %s", word)
                        count_skipped += 1
                    else:
                        indices.append((s_idx, w_idx, lemma))
                    sent.append(word)
                sentences.append(sent)
            orig_sentences  = [[w for w in s] for s in sentences]
            for s_idx, w_idx, lemma in indices:
                replacements = replace_target_word(sentences[s_idx], w_idx)
                replacements_sents = [t[0] for t in replacements]
                sense_i = choose_sense(
                    sentences, s_idx, replacements_sents,
                    embed_func=sif_embeds,
                    distance_func=scipy.spatial.distance.minkowski)
                if sense_i is None:
                    continue
                pprint.pprint([detok_sent(sent) for sent in orig_sentences])
                pprint.pprint((s_idx, w_idx, orig_sentences[s_idx][w_idx]))
                print("Correct sense:", lemma)
                print("Correct sense:", lemma.synset())
                print("Correct sense:", lemma.synset().definition())

                predicted_synset = replacements[sense_i][1]
                print("Predicted sense:", predicted_synset)
                print("Predicted sense:", predicted_synset.definition())
                print(detok_sent(replacements[sense_i][0]))
                print("** ** " * 16)
                if lemma.synset() == predicted_synset:
                    count_correct += 1
                else:
                    count_wrong += 1

        print("Total correct", count_correct)
        print("Total wrong", count_wrong)
        print("Total skipped", count_skipped)
