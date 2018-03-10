#!/usr/bin/env python
import nltk
import sklearn
import sklearn.cluster
import numpy as np

wn = nltk.wordnet.wordnet


def distance_metric(s1, s2):
    # https://stackoverflow.com/a/20799567/2694385
    # sim1, sim2 = wn.path_similarity(s1, s2), wn.path_similarity(s2, s1)
    # if sim1 is None:
        # sim = sim2
    # elif sim2 is None:
        # sim = sim1
    # else:
        # sim = max(sim1, sim2)
    # distance = 1 + (1 - sim)

    # distance = s1.shortest_path_distance(s2)
    # if distance is None:
        # distance = s2.shortest_path_distance(s1)
    # if distance is None:
        # distance = np.nan

    wup = wn.wup_similarity(s1, s2)
    if wup is None:
        wup = wn.wup_similarity(s2, s1)
    if wup is None:
        distance = np.nan
    else:
        distance = 1 + 1 - wup

    return distance



def pairwise_dist(synsets):
    dist = np.zeros((len(synsets), len(synsets)))
    for i, s1 in enumerate(synsets):
        for j, s2 in enumerate(synsets):
            dist[i, j] = distance_metric(s1, s2)
    max_dist = np.max(dist[~np.isnan(dist)])
    dist[np.isnan(dist)] = max_dist + 1
    return dist


def cluster(synsets):
    dist = pairwise_dist(synsets)
    upper = np.triu(dist)
    nonzero_upper = upper[upper > 0]
    eps = np.percentile(nonzero_upper, 10)
    print("EPS:", eps)
    clusterer = sklearn.cluster.DBSCAN(
        eps=eps,
        min_samples=1,
        metric='precomputed')
    fit = clusterer.fit(dist)
    clusters = dict()
    counter = -1
    for label, synset in zip(fit.labels_, synsets):
        if label < 0:
            label = counter
            counter -= 1
        if label in clusters:
            clusters[label].add(synset)
        else:
            clusters[label] = set((synset,))
    return list(clusters.values())



if __name__ == '__main__':
    import pprint
    synsets = wn.synsets('bank')
    pprint.pprint(synsets)
    pprint.pprint(pairwise_dist(synsets))
    # pprint.pprint([(i, ss, ss.definition()) for i, ss in enumerate(synsets)])
    pprint.pprint([
            set([(x, x.definition()) for x in group])
            for group in cluster(synsets)
        ])
