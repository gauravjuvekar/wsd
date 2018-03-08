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

import logging
logging.basicConfig(loglevel=logging.DEBUG)
log = logging.getLogger(__name__)

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


def sif_embeds(sent_list):
    idx_mat, weight_mat, data = SIF.data_io.prepare_data(sent_list, sif_db)
    params = SIF.params.params()
    params.rmpc = 1
    embedding = SIF.SIF_embedding.SIF_embedding(idx_mat,
                                                weight_mat,
                                                data,
                                                params)
    return list(embedding)



@functools.lru_cache()
def detok_sent(sent):
    detokenizer = nltk.tokenize.moses.MosesDetokenizer()
    return detokenizer.detokenize(sent, return_str=True)

# @functools.lru_cache()
# def detok_sent(sent):
    # return ' '.join(sent)


@functools.lru_cache()
def s2v_embed_wrapped(sent):
    return s2v_model.embed_sentence(sent)


def s2v_embeds(sents):
    return [s2v_embed_wrapped(detok_sent(sent)) for sent in sents]


def get_replacements(tok_sent, index, lemma, pos=None):
    # given a sentence represented as a list of tokens, and the index of the
    # token to be replaced (i.e the token to be disambiguated), return list of
    # sentences with hypernym replaced for each sense of target token word
    lemset = set()
    for synset in wordnet.synsets(lemma, pos=pos):
        hypernyms = synset.hypernyms()
        if not hypernyms:
            log.warn("Synset %s has no hypernyms", synset)
        for hypernym in hypernyms:
            for lem in hypernym.lemmas():
                lemset.add((synset, lem))

    sent_list = []
    for synset, lem in lemset:
        elem_list = tuple([word
            for word in nltk.tokenize.word_tokenize(
                lem.name().replace('_', ' '))
            ])
        new_sent = tok_sent[:index] + elem_list + tok_sent[index + 1:]
        yield (new_sent, synset)


def choose_sense(sentences, index_to_replace, replacements,
                 embed_func, distance_func):

    average_dist = []
    s_idx = index_to_replace
    for replacement in replacements:
        replaced_para = sentences[:s_idx] + replacement + sentences[s_idx + 1:]
        embeds = embed_func(replaced_para)
        replaced_embed = embeds[s_idx]
        pairwise_dist = [distance_func(replaced_embed, context_embed)
                         for context_embed in embeds]
        # distance with itself will be 0
        average_dist.append(sum(pairwise_dist) / (len(pairwise_dist) - 1))

    if len(average_dist):
        return [idx for idx, dist in sorted(enumerate(average_dist),
                                            key=lambda x: x[1])]

def eval_semcor(paras):
    count_correct = 0
    count_wrong = 0
    count_skipped = 0

    for para in paras:
        sentences = []
        indices = []
        for s_idx, sentence in enumerate(para):
            sent = []
            w_idx = 0
            for w_group_idx, word in enumerate(sentence):
                if word['true_sense'] is None:
                    # Don't need to disambiguate this word
                    pass
                elif isinstance(word['true_sense'], str):
                    # Should be disambiguated, but we couldn't find it's lemma
                    # in wordnet
                    log.warn("No lemma found for %s", word)
                    count_skipped += 1
                else:
                    # Disambiguate this
                    indices.append({'s_idx': s_idx,
                                    'w_idx': w_idx,
                                    'w_group_idx': w_group_idx,
                                    'sense': word['true_sense'],
                                    'lemma': word['lemma'],
                                    'pos': word['pos']})
                sent.extend(word['words'])
                w_idx += len(word['words'])
            sentences.append(tuple(sent))
        sentences = tuple(sentences)
        orig_sentences  = sentences
        for word in indices:
            replacements = list(get_replacements(sentences[word['s_idx']],
                                                 word['w_idx'],
                                                 word['lemma'],
                                                 word['pos']))
            replacements_sents = [t[0] for t in replacements]
            sense_order = choose_sense(
                sentences, s_idx, replacements_sents,
                embed_func=sif_embeds,
                distance_func=scipy.spatial.distance.minkowski)
            if not sense_order:
                log.warn("No sense order obtained")
                count_skipped += 1
                continue

            senses = [replacements[i][1] for i in sense_order]
            pprint.pprint([detok_sent(sent) for sent in orig_sentences])
            pprint.pprint(word)
            true_sense = word['sense']
            print("Correct sense:", true_sense)
            print("Correct sense:", true_sense.synset())
            print("Correct sense:", true_sense.synset().definition())

            print("Predicted:")
            predicted_synset = senses[0]
            pprint.pprint([(sense, sense.definition()) for sense in senses])
            print(detok_sent(replacements[sense_order[0]][0]))
            print("*" * 80)
            if true_sense.synset() == predicted_synset:
                count_correct += 1
            else:
                count_wrong += 1

    print("Total correct", count_correct)
    print("Total wrong", count_wrong)
    print("Total skipped", count_skipped)



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
            paras = semcor_reader.read_semcor(f)

        eval_semcor(paras)

