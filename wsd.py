#!/usr/bin/env python
import SIF
import sent2vec
import semcor_reader
import cluster_wordnet
import baseline

import scipy
import scipy.spatial
import scipy.stats
import scipy.spatial.distance

import nltk
import nltk.tokenize
import nltk.tokenize.moses

import pywsd

import functools
import statistics
from collections import defaultdict

import numpy
numpy.set_printoptions(threshold=10)

import os

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


def underscore_tokenize(sentence):
    return tuple(
        [word for word in
            nltk.tokenize.word_tokenize(sentence.replace('_', ' '))
            ])


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
            elem_list = underscore_tokenize(synset.definition())
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
        elem_list = underscore_tokenize(lem.name())
        new_sent = tok_sent[:index] + elem_list + tok_sent[index + 1:]
        yield new_sent, synset



def choose_sense(sentences, target_word, embed_func, distance_func):
    replacements = list(
        get_replacements(
            sentences[target_word['s_idx']],
            target_word['w_idx'],
            target_word['lemma'],
            target_word['pos']))

    average_dist = []
    s_idx = target_word['s_idx']
    synset_dist = defaultdict(set)

    for new_sent, synset in replacements:
        replaced_para = sentences[:s_idx] + new_sent + sentences[s_idx + 1:]
        embeds = embed_func(replaced_para)
        replaced_embed = embeds[s_idx]
        pairwise_dist = [distance_func(replaced_embed, context_embed)
                         for context_embed in embeds]
        # distance with itself will be 0
        this_distance = sum(pairwise_dist) / (len(pairwise_dist) - 1)
        average_dist.append(this_distance)
        synset_dist[synset].add(this_distance)

    # for synset, dist_set in synset_dist.items():
        # synset_dist[synset] = sum(dist_set) / len(dist_set)

    for synset, dist_set in synset_dist.items():
        synset_dist[synset] = min(dist_set)

    return [
        (synset, {'dist': dist})
        for synset, dist in sorted(synset_dist.items(), key=lambda x:x[1])]


def choose_sense_nocontext_double_sort(
        sentences, target_word, embed_func, distance_func):
    replacements = list(
        get_replacements(
            sentences[target_word['s_idx']],
            target_word['w_idx'],
            target_word['lemma'],
            target_word['pos']))

    s_idx = target_word['s_idx']
    dist = []
    synset_dist = defaultdict(list)
    for new_sent, synset in replacements:
        orig_sent = sentences[s_idx]
        embeds = embed_func((orig_sent, new_sent))
        this_distance = distance_func(embeds[0], embeds[1])
        cosine_dist = scipy.spatial.distance.cosine(embeds[0], embeds[1])

        append_dict = {'dist': this_distance,
                       'cosine_dist': cosine_dist,
                       'embedding': embeds[1]}
        synset_dist[synset].append(append_dict)

    for synset, dist in synset_dist.items():
        synset_dist[synset] = min(dist, key=lambda x: x['dist'])

    sort_1 = list(sorted(synset_dist.items(), key=lambda x:x[1]['dist']))
    high_idx = max(1, len(sort_1) // 2 + 1)
    sort_2 = (
        list(sorted(sort_1[:high_idx], key=lambda x:x[1]['cosine_dist'])) +
        list(sort_1[high_idx:]))
    return sort_2


def choose_sense_definition(
        sentences, target_word, embed_func, distance_func):
    orig_sent = sentences[target_word['s_idx']]
    synset_dist = defaultdict(list)
    for synset in wordnet.synsets(target_word['lemma'], target_word['pos']):
        definition = underscore_tokenize(synset.definition())
        embeds = embed_func((orig_sent, definition))
        distance = distance_func(embeds[0], embeds[1])
        append_dict = {'dist': distance, 'embedding': embeds[1]}
        synset_dist[synset].append(append_dict)

    for synset, dist in synset_dist.items():
        synset_dist[synset] = min(dist, key=lambda x: x['dist'])
    return list(sorted(synset_dist.items(), key=lambda x:x[1]['dist']))


def eval_semcor(paras, stats=None):
    if stats is None:
        stats = defaultdict(int)
    # {
        # 'baseline_first': 0,
        # 'same_cluster_baseline_first': 0,
        # 'baseline_random': 0,
        # 'same_cluster_baseline_random': 0,
        # 'baseline_most_frequent': 0,
        # 'same_cluster_baseline_most_frequent': 0,
        # 'nocontext_double_sort': 0,
        # 'same_cluster_nocontext_double_sort': 0,
        # 'context_sentences': 0,
        # 'same_cluster_context_sentences': 0,
        # 'definition': 0,
        # 'same_cluster_definition': 0,
        # 'total': 0,
    # }

    for para in paras:
        sentences = []
        indices = []
        for s_idx, sentence in enumerate(para):
            sent = []
            w_idx = 0
            for w_group_idx, word in enumerate(sentence):
                if word['true_senses'] is None:
                    # Don't need to disambiguate this word
                    pass
                else:
                    valid_senses = []
                    for sense in word['true_senses']:
                        if isinstance(sense, str):
                            # Should be disambiguated, but we couldn't find
                            # it's lemma in wordnet
                            log.warn("No lemma found for %s", word)
                            stats['total_skipped'] += 1
                        else:
                            valid_senses.append(sense)
                    if valid_senses:
                        # Disambiguate this
                        indices.append({'s_idx': s_idx,
                                        'w_idx': w_idx,
                                        'w_group_idx': w_group_idx,
                                        'senses': valid_senses,
                                        'lemma': word['lemma'],
                                        'pos': word['pos']})
                sent.extend(word['words'])
                w_idx += len(word['words'])
            sentences.append(tuple(sent))
        sentences = tuple(sentences)
        orig_sentences  = sentences
        for word in indices:
            sense_output = dict()
            sense_output['baseline_first'] = baseline.first_sense(
                word['lemma'], word['pos'])
            sense_output['baseline_random'] = baseline.random_sense(
                word['lemma'], word['pos'])
            sense_output['baseline_most_frequent'] = (
                baseline.max_lemma_count_sense(word['lemma'], word['pos']))
            sense_output['nocontext_double_sort'] = (
                choose_sense_nocontext_double_sort(
                    sentences,
                    target_word=word,
                    embed_func=sif_embeds,
                    distance_func=scipy.spatial.distance.sqeuclidean))
            sense_output['context_sentences'] = (
                choose_sense(
                    sentences,
                    target_word=word,
                    embed_func=sif_embeds,
                    distance_func=scipy.spatial.distance.cosine))
            sense_output['definition'] = choose_sense_definition(
                sentences,
                target_word=word,
                embed_func=sif_embeds,
                distance_func=scipy.spatial.distance.sqeuclidean)

            true_senses = [sense.synset() for sense in word['senses']]
            clustered_senses = cluster_wordnet.cluster(
                wordnet.synsets(word['lemma'], word['pos']))
            stats['total'] += 1
            for method, result in sense_output.items():
                if not result:
                    log.warn('No result for %s', method)
                    continue
                predicted_sense = result[0][0]
                if predicted_sense in true_senses:
                    stats[method] += 1
                for cluster in clustered_senses:
                    if (any(x in cluster for x in true_senses) and
                            predicted_sense in cluster):
                        stats['same_cluster_' + method] += 1
                        break


            pprint.pprint([detok_sent(sent) for sent in orig_sentences])
            pprint.pprint(word)
            print("Correct senses:", [x.definition() for x in true_senses])

            print("Predicted:")
            pprint.pprint(sense_output)

            print("*" * 80)

    pprint.pprint(stats)
    return stats



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

    if False:
        semcor_file = './data/datasets/semcor3.0/brownv/tagfiles/br-r01'
        with open(semcor_file, 'rb') as f:
            paras = semcor_reader.read_semcor(f)

        eval_semcor(paras)

    if True:
        combined_stats = defaultdict(int)
        brown_dir = './data/datasets/semcor3.0/brownv/tagfiles'
        for f in os.listdir(brown_dir):
            with open(os.path.join(brown_dir, f), 'rb') as f:
                paras = semcor_reader.read_semcor(f)
            stats = eval_semcor(paras)
            for k, v in stats.items():
                combined_stats[k] += v
        pprint.pprint(combined_stats)
