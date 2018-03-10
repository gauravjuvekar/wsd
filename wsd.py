#!/usr/bin/env python
import SIF
import sent2vec
import semcor_reader
import scipy
import scipy.spatial
import scipy.stats
import scipy.spatial.distance

import nltk
import nltk.tokenize
import nltk.tokenize.moses

import functools
import statistics

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

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
        hyponyms = synset.hyponyms()
        if not hypernyms:
            log.info("Synset %s has no hypernyms", synset)
        if not hyponyms:
            log.info("Synset %s has no hyponyms", synset)
        if not hypernyms and not hyponyms:
            definition = nltk.tokenize.word_tokenize(synset.definition())
            elem_list = tuple([word for word in definition])
            new_sent = tok_sent[:index] + elem_list + tok_sent[index + 1:]
            yield new_sent, synset


        for hypernym in hypernyms:
            for lem in hypernym.lemmas():
                lemset.add((synset, lem))
        for hyponym in hyponyms:
            for lem in hyponym.lemmas():
                lemset.add((synset, lem))


    sent_list = []
    for synset, lem in lemset:
        elem_list = tuple([word
            for word in nltk.tokenize.word_tokenize(
                lem.name().replace('_', ' '))
            ])
        new_sent = tok_sent[:index] + elem_list + tok_sent[index + 1:]
        yield new_sent, synset



def choose_sense(sentences, index_to_replace, replacements,
                 embed_func, distance_func):
    average_dist = []
    s_idx = index_to_replace
    synset_dist = dict()
    for new_sent, synset in replacements:
        replaced_para = sentences[:s_idx] + new_sent + sentences[s_idx + 1:]
        embeds = embed_func(replaced_para)
        replaced_embed = embeds[s_idx]
        pairwise_dist = [distance_func(replaced_embed, context_embed)
                         for context_embed in embeds]
        # distance with itself will be 0
        this_distance = sum(pairwise_dist) / (len(pairwise_dist) - 1)
        average_dist.append(this_distance)
        if synset in synset_dist:
            synset_dist[synset].add(this_distance)
        else:
            synset_dist[synset] = set((this_distance,))

    # for synset, dist_set in synset_dist.items():
        # synset_dist[synset] = sum(dist_set) / len(dist_set)

    for synset, dist_set in synset_dist.items():
        synset_dist[synset] = min(dist_set)

    return list(sorted(synset_dist.items(), key=lambda x:x[1]))


def choose_sense_nocontext(sentences, index_to_replace, replacements,
                           embed_func, distance_func):
    s_idx = index_to_replace
    dist = []
    synset_dist = dict()
    for new_sent, synset in replacements:
        orig_sent = sentences[index_to_replace]
        embeds = embed_func((orig_sent, new_sent))
        this_distance = distance_func(embeds[0], embeds[1])
        cosine_dist = scipy.spatial.distance.cosine(embeds[0], embeds[1])
        if synset in synset_dist:
            synset_dist[synset].add(
                (this_distance, cosine_dist, tuple(embeds[1])))
        else:
            synset_dist[synset] = set(
                ((this_distance, cosine_dist, tuple(embeds[1])),))

    for synset, dist_set in synset_dist.items():
        synset_dist[synset] = min(dist_set, key=lambda x: x[0])

    sort_1 = list(sorted(synset_dist.items(), key=lambda x:x[1][0]))
    high_idx = max(1, len(sort_1) // 2 + 1)
    sort_2 = (list(sorted(sort_1[:high_idx], key=lambda x:x[1][1])) +
              list(sort_1[high_idx:]))
    return sort_2


def eval_semcor(paras):
    count_correct = 0
    count_wrong = 0
    count_skipped = 0
    count_rank_none = 0
    rank_list = []

    n_words = 0
    n_senses = 0
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
            n_senses += len(wordnet.synsets(word['lemma'], word['pos']))
            n_words += 1
            replacements = list(get_replacements(sentences[word['s_idx']],
                                                 word['w_idx'],
                                                 word['lemma'],
                                                 word['pos']))
            sense_order = choose_sense_nocontext(
                sentences, s_idx, replacements,
                embed_func=sif_embeds,
                distance_func=scipy.spatial.distance.sqeuclidean)
            if not sense_order:
                log.warn("No sense order obtained")
                count_skipped += 1
                continue

            clustered_senses =

            pprint.pprint([detok_sent(sent) for sent in orig_sentences])
            pprint.pprint(word)
            true_sense = word['sense']
            print("Correct sense:", true_sense)
            print("Correct sense:", true_sense.synset().definition())

            print("Predicted:")
            predicted_synset = sense_order[0][0]
            pprint.pprint([((sense, dist[0]), sense.definition())
                           for sense, dist in sense_order])
            try:
                rank = [sense for sense, dist in sense_order
                        ].index(true_sense.synset())
            except ValueError:
                rank = None
                count_rank_none += 1
            else:
                rank_list.append(rank)

            print("Rank:", rank)
            if rank == 1 and len(sense_order) >= 2:
                print('0-1 distance',
                    "Euclidean",
                    scipy.spatial.distance.euclidean(
                        sense_order[0][1][2], sense_order[1][1][2]),
                    "Cosine",
                    scipy.spatial.distance.cosine(
                        sense_order[0][1][2], sense_order[1][1][2]))
            print("*" * 80)


            if true_sense.synset() == predicted_synset:
                count_correct += 1
            else:
                count_wrong += 1

    print("Total correct", count_correct)
    print("Total wrong", count_wrong)
    print("Total skipped", count_skipped)
    print("Total rank None", count_rank_none)

    print("Mean rank", statistics.mean(rank_list))
    print("Median rank", statistics.median_grouped(rank_list))

    historgram = scipy.stats.itemfreq(rank_list)
    print("Rank histogram")
    pprint.pprint(historgram)

    print("Avg senses per word", n_senses / n_words)




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

