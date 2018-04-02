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

import pickle

# import pywsd

import functools
import statistics
from collections import defaultdict

import numpy
numpy.set_printoptions(threshold=10)

import os

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


wordnet = nltk.wordnet.wordnet
wordnet.ensure_loaded()

ENABLE = (
    'sif',
    # 's2v',
    )

PREFETCH = True

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
    return [s2v_embed_wrapped(detok_sent(tuple(sent))) for sent in sents]


def underscore_tokenize(sentence):
    return tuple(
        [word for word in
            nltk.tokenize.word_tokenize(sentence.replace('_', ' '))
            ])


def get_replacements(tok_sent, index, synsets, lemma, pos=None):
    # given a sentence represented as a list of tokens, and the index of the
    # token to be replaced (i.e the token to be disambiguated), return list of
    # sentences with hypernym replaced for each sense of target token word
    lemset = set()
    for synset in synsets:
        hypernyms = synset.hypernyms()
        hyponyms = synset.hyponyms()
        if not hypernyms:
            log.debug("Synset %s has no hypernyms", synset)
        if not hyponyms:
            log.debug("Synset %s has no hyponyms", synset)
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



def choose_sense(sentences, target_word, senses, embed_func, distance_func):
    replacements = list(
        get_replacements(
            sentences[target_word['s_idx']],
            target_word['w_idx'],
            synsets=senses,
            lemma=target_word['lemma'],
            pos=target_word['pos']))

    if PREFETCH:
        embed_func(tuple(sent for sent, _ in replacements))

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


def choose_sense_multiply_dist(
        sentences, target_word, senses, embed_func, distance_func):
    replacements = list(
        get_replacements(
            sentences[target_word['s_idx']],
            target_word['w_idx'],
            synsets=senses,
            lemma=target_word['lemma'],
            pos=target_word['pos']))

    if PREFETCH:
        embed_func(tuple(sent for sent, _ in replacements))

    average_dist = []
    s_idx = target_word['s_idx']
    synset_dist = defaultdict(set)

    for new_sent, synset in replacements:
        replaced_para = sentences[:s_idx] + new_sent + sentences[s_idx + 1:]
        embeds = embed_func(replaced_para)
        replaced_embed = embeds[s_idx]
        pairwise_dist = [distance_func(replaced_embed, context_embed)
                         for context_embed in embeds
                         if context_embed is not replaced_embed]
        # distance with itself will be 0
        product = functools.reduce(lambda x, y: x * y, pairwise_dist)
        average_dist.append(product)
        synset_dist[synset].add(product)

    # for synset, dist_set in synset_dist.items():
        # synset_dist[synset] = sum(dist_set) / len(dist_set)

    for synset, dist_set in synset_dist.items():
        synset_dist[synset] = min(dist_set)

    return [
        (synset, {'dist': dist})
        for synset, dist in sorted(synset_dist.items(), key=lambda x:x[1])]



def choose_sense_nocontext_double_sort(
        sentences, target_word, senses, embed_func, distance_func):
    replacements = list(
        get_replacements(
            sentences[target_word['s_idx']],
            target_word['w_idx'],
            synsets=senses,
            lemma=target_word['lemma'],
            pos=target_word['pos']))

    if PREFETCH:
        embed_func(tuple(sent for sent, _ in replacements))

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
        sentences, target_word, senses, embed_func, distance_func):
    orig_sent = sentences[target_word['s_idx']]
    synset_dist = defaultdict(list)
    for synset in senses:
        definition = underscore_tokenize(synset.definition())
        embeds = embed_func((orig_sent, definition))
        distance = distance_func(embeds[0], embeds[1])
        append_dict = {'dist': distance, 'embedding': embeds[1]}
        synset_dist[synset].append(append_dict)

    for synset, dist in synset_dist.items():
        synset_dist[synset] = min(dist, key=lambda x: x['dist'])
    return list(sorted(synset_dist.items(), key=lambda x:x[1]['dist']))


def choose_sense_examples(
        sentences, target_word, senses, embed_func, distance_func):
    orig_sent = sentences[target_word['s_idx']]

    synset_dist = defaultdict(list)
    for synset in senses:
        examples = synset.examples()
        if not len(examples):
            log.debug("Synset %s has no examples", synset)
            continue
        else:
            for example in examples:
                example = underscore_tokenize(example)
                embeds = embed_func((orig_sent, example))
                distance = distance_func(embeds[0], embeds[1])
                append_dict = {'dist': distance, 'embedding': embeds[1]}
                synset_dist[synset].append(append_dict)

    for synset, dist in synset_dist.items():
        synset_dist[synset] = min(dist, key=lambda x: x['dist'])
    return list(sorted(synset_dist.items(), key=lambda x:x[1]['dist']))


def choose_sense_5word_definition(
        sentences, target_word, senses, embed_func, distance_func):
    flat = [word for sentence in sentences for word in sentence]
    target_idx = (sum(len(s) for s in sentences[:target_word['s_idx']]) +
                  target_word['w_idx'])
    low_idx = max(0, target_idx - 2)
    orig_context = flat[low_idx:low_idx + 5 + target_word['w_group_len'] - 1]

    synset_dist = defaultdict(list)
    for synset in senses:
        definition = underscore_tokenize(synset.definition())
        embeds = embed_func((orig_context, definition))
        distance = distance_func(embeds[0], embeds[1])
        append_dict = {'dist': distance, 'embedding': embeds[1]}
        synset_dist[synset].append(append_dict)

    for synset, dist in synset_dist.items():
        synset_dist[synset] = min(dist, key=lambda x: x['dist'])
    return list(sorted(synset_dist.items(), key=lambda x:x[1]['dist']))


def choose_sense_7word_definition(
        sentences, target_word, senses, embed_func, distance_func):
    flat = [word for sentence in sentences for word in sentence]
    target_idx = (sum(len(s) for s in sentences[:target_word['s_idx']]) +
                  target_word['w_idx'])
    low_idx = max(0, target_idx - 2)
    orig_context = flat[low_idx:low_idx + 7 + target_word['w_group_len'] - 1]

    synset_dist = defaultdict(list)
    for synset in senses:
        definition = underscore_tokenize(synset.definition())
        embeds = embed_func((orig_context, definition))
        distance = distance_func(embeds[0], embeds[1])
        append_dict = {'dist': distance, 'embedding': embeds[1]}
        synset_dist[synset].append(append_dict)

    for synset, dist in synset_dist.items():
        synset_dist[synset] = min(dist, key=lambda x: x['dist'])
    return list(sorted(synset_dist.items(), key=lambda x:x[1]['dist']))


def choose_sense_5word_examples(
        sentences, target_word, senses, embed_func, distance_func):
    flat = [word for sentence in sentences for word in sentence]
    target_idx = (sum(len(s) for s in sentences[:target_word['s_idx']]) +
                  target_word['w_idx'])
    low_idx = max(0, target_idx - 2)
    orig_context = flat[low_idx:low_idx + 5 + target_word['w_group_len'] - 1]

    synset_dist = defaultdict(list)
    for synset in senses:
        examples = synset.examples()
        if not len(examples):
            log.warn("Synset %s has no examples", synset)
            continue
        else:
            for example in examples:
                example = underscore_tokenize(example)
                embeds = embed_func((orig_context, example))
                distance = distance_func(embeds[0], embeds[1])
                append_dict = {'dist': distance, 'embedding': embeds[1]}
                synset_dist[synset].append(append_dict)

    for synset, dist in synset_dist.items():
        synset_dist[synset] = min(dist, key=lambda x: x['dist'])
    return list(sorted(synset_dist.items(), key=lambda x:x[1]['dist']))



def choose_sense_7word_examples(
        sentences, target_word, senses, embed_func, distance_func):
    flat = [word for sentence in sentences for word in sentence]
    target_idx = (sum(len(s) for s in sentences[:target_word['s_idx']]) +
                  target_word['w_idx'])
    low_idx = max(0, target_idx - 2)
    orig_context = flat[low_idx:low_idx + 7 + target_word['w_group_len'] - 1]

    synset_dist = defaultdict(list)
    for synset in senses:
        examples = synset.examples()
        if not len(examples):
            log.warn("Synset %s has no examples", synset)
            continue
        else:
            for example in examples:
                example = underscore_tokenize(example)
                embeds = embed_func((orig_context, example))
                distance = distance_func(embeds[0], embeds[1])
                append_dict = {'dist': distance, 'embedding': embeds[1]}
                synset_dist[synset].append(append_dict)

    for synset, dist in synset_dist.items():
        synset_dist[synset] = min(dist, key=lambda x: x['dist'])
    return list(sorted(synset_dist.items(), key=lambda x:x[1]['dist']))



with open('semcor_stats.pickle', 'rb') as f:
    corpus_stats = pickle.load(f)
    corpus_total_lemma_count = sum(corpus_stats['senses'].values())
    for k, v in list(corpus_stats['senses'].items()):
        k2 = wordnet.lemma_from_key(k).synset()
        corpus_stats['senses'][k2] = corpus_stats['senses'][k]
        del corpus_stats['senses'][k]


def choose_sense_weighted(
        sentences, target_word, senses, embed_func, distance_func):
    unweighted = choose_sense(
        sentences, target_word, senses, embed_func, distance_func)
    sum_1_dist = 0
    for synset, v in unweighted:
        sum_1_dist += v['dist']

    sum_1_dist = 1 / sum_1_dist
    for synset, v in unweighted:
        v['probability'] = (1 / v['dist']) / sum_1_dist
        v['weighted'] = v['probability'] * (corpus_stats['senses'][synset] /
                                            corpus_total_lemma_count)
    return list(sorted(unweighted,
                       key=lambda x: x[1]['weighted'],
                       reverse=True))


def get_sense_outputs(sentences, synsets, target_word, embed_func):
    sense_output = dict()

    sense_output['baseline_first'] = baseline.choose_first(synsets)
    sense_output['baseline_random'] = baseline.choose_random(synsets)
    sense_output['baseline_most_frequent'] = (
        baseline.choose_max_lemma_count(synsets))
    sense_output['nocontext_double_sort'] = (
        choose_sense_nocontext_double_sort(
            sentences,
            target_word=target_word,
            senses=synsets,
            embed_func=embed_func,
            distance_func=scipy.spatial.distance.euclidean))
    sense_output['context_sentences'] = (
        choose_sense(
            sentences,
            target_word=target_word,
            senses=synsets,
            embed_func=embed_func,
            distance_func=scipy.spatial.distance.euclidean))
    sense_output['multiply_dist_cosine'] = choose_sense_multiply_dist(
            sentences,
            target_word=target_word,
            senses=synsets,
            embed_func=embed_func,
            distance_func=scipy.spatial.distance.cosine)
    sense_output['definition'] = choose_sense_definition(
        sentences,
        target_word=target_word,
        senses=synsets,
        embed_func=embed_func,
        distance_func=scipy.spatial.distance.euclidean)
    sense_output['examples'] = choose_sense_examples(
        sentences,
        target_word=target_word,
        senses=synsets,
        embed_func=embed_func,
        distance_func=scipy.spatial.distance.euclidean)
    sense_output['7word_examples'] = choose_sense_7word_examples(
        sentences,
        target_word=target_word,
        senses=synsets,
        embed_func=embed_func,
        distance_func=scipy.spatial.distance.euclidean)
    sense_output['5word_examples'] = choose_sense_5word_examples(
        sentences,
        target_word=target_word,
        senses=synsets,
        embed_func=embed_func,
        distance_func=scipy.spatial.distance.euclidean)
    sense_output['5word_definition'] = choose_sense_5word_definition(
        sentences,
        target_word=target_word,
        senses=synsets,
        embed_func=embed_func,
        distance_func=scipy.spatial.distance.euclidean)
    sense_output['7word_definition'] = choose_sense_7word_definition(
        sentences,
        target_word=target_word,
        senses=synsets,
        embed_func=embed_func,
        distance_func=scipy.spatial.distance.euclidean)
    sense_output['weighted'] = choose_sense_weighted(
        sentences,
        target_word=target_word,
        senses=synsets,
        embed_func=embed_func,
        distance_func=scipy.spatial.distance.sqeuclidean)
    return sense_output


def reduce_stats(iterator, stats=None):
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

    for d in iterator:
        pprint.pprint(d)
        true_senses = [sense.synset() for sense in d['word']['senses']]
        clustered_senses = cluster_wordnet.cluster(d['synsets'])
        pos = d['word']['pos']

        stats['total'] += 1
        stats['total_pos_' + pos] += 1

        for method, result in d['outputs'].items():
            if not result:
                log.warn('No result for %s', method)
                stats['no_result'] += 1
                stats['no_result_pos_' + pos] += 1
                continue

            predicted_sense = result[0][0]
            if predicted_sense in true_senses:
                stats[method] += 1
                stats[method + '_pos_' + pos] += 1
            for cluster in clustered_senses:
                if (any(x in cluster for x in true_senses) and
                        predicted_sense in cluster):
                    stats['same_cluster_' + method] += 1
                    stats['same_cluster_' + method + '_pos_' + pos] += 1
                    break
        print("*" * 80)
    return stats


def eval_semcor(paras, embed_func, stats=None):
    if stats is None:
        stats = defaultdict(int)
    for para_idx, para in enumerate(paras):
        log.info("Para: %d", para_idx)
        sentences = []
        indices = []
        for s_idx, sentence in enumerate(para):
            sent = []
            w_idx = 0
            for w_group_idx, word in enumerate(sentence):
                if word['disambiguate?']:
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
                        word['s_idx'] = s_idx
                        word['w_idx'] = w_idx
                        word['w_group_idx'] = w_group_idx
                        word['w_group_len'] = len(word['words'])
                        word['senses'] = valid_senses
                        indices.append(word)
                sent.extend(word['words'])
                w_idx += len(word['words'])
            sentences.append(tuple(sent))

        sentences = tuple(sentences)
        orig_sentences  = sentences
        for disambiguate_idx, word in enumerate(indices):
            log.info("Para: %d, disambiguate_idx: %d/%d",
                     para_idx, disambiguate_idx, len(indices))
            synsets = wordnet.synsets(word['lemma'], word['pos'])

            sense_output = get_sense_outputs(
                sentences=sentences,
                target_word=word,
                synsets=synsets,
                embed_func=embed_func)

            pprint.pprint([detok_sent(sent) for sent in orig_sentences])
            pprint.pprint(word)
            yield {
                'word': word,
                'synsets': synsets,
                'outputs': sense_output
                }



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

        pprint.pprint(list(eval_semcor(paras)))

    if True:
        combined_stats = defaultdict(int)
        brown_dir = './data/datasets/semcor3.0/brown1/tagfiles'
        count = 0
        for f in os.listdir(brown_dir):
            count += 1
            if count > 1: break
            with open(os.path.join(brown_dir, f), 'rb') as f:
                paras = semcor_reader.read_semcor(f)
            stats = reduce_stats(eval_semcor(paras, embed_func=sif_embeds))
            for k, v in stats.items():
                combined_stats[k] += v
        pprint.pprint(combined_stats)

    if False:
        brownv_dir = './data/datasets/semcor3.0/brownv/tagfiles'
        brown1_dir = './data/datasets/semcor3.0/brown1/tagfiles'
        brown2_dir = './data/datasets/semcor3.0/brown2/tagfiles'
        stats = {
            'pos': defaultdict(int),
            'senses': defaultdict(int),
            'total_skipped': 0,
            'total_count': 0
            }
        file_count = 0
        for d in (brown1_dir, brown2_dir, brownv_dir):
            for f in os.listdir(d):
                file_count += 1
                print(file_count, d, f)

                with open(os.path.join(d, f), 'rb') as f:
                    paras = semcor_reader.read_semcor(f)

                for para in paras:
                    for sentence in para:
                        for word in sentence:
                            if word['true_senses'] is None:
                                # Don't need to disambiguate this word
                                pass
                            else:
                                stats['total_count'] += 1
                                pos = word['pos']
                                stats['pos'][pos] += 1
                                for sense in word['true_senses']:
                                    if isinstance(sense, str):
                                        # Should be disambiguated, but we
                                        # couldn't find it's lemma in wordnet
                                        log.warn("No lemma found for %s", word)
                                        stats['total_skipped'] += 1
                                    else:
                                        stats['senses'][sense.key()] += 1
        import pickle
        with open('semcor_stats.pickle', 'wb') as f:
            pickle.dump(stats, f)
        pprint.pprint(stats)
