#!/usr/bin/env python
import pywsd
import nltk
wn = nltk.wordnet.wordnet
wn.ensure_loaded()

def first_sense(lemma, pos):
    return [(pywsd.baseline.first_sense(lemma, pos),)]

def random_sense(lemma, pos):
    return [(pywsd.baseline.random_sense(lemma, pos),)]

# define function manually pending  https://github.com/alvations/pywsd/pull/38
def max_lemma_count_sense(ambiguous_word, pos=None):
    """
    Returns the sense with the highest lemma_name count.
    The max_lemma_count() can be treated as a rough gauge for the
    Most Frequent Sense (MFS), if no other sense annotated corpus is available.
    NOTE: The lemma counts are from the Brown Corpus
    """
    try: sense2lemmacounts = {i:sum(j.count() for j in i.lemmas()) \
                              for i in wn.synsets(ambiguous_word, pos=None)}
    except: sense2lemmacounts = {i:sum(j.count() for j in i.lemmas) \
                                 for i in wn.synsets(ambiguous_word, pos=None)}
    return [(max(sense2lemmacounts, key=sense2lemmacounts.get),)]
