"""
Experiment script for Purver, Sadrzadeh, Kempson, Wijnholds, Hough "Incremental Composition in Distributional Semantics"
Copyright Matthew Purver, QMUL 2019
"""

import csv
import os
import pickle
import random
import re
import shelve

import numpy as np
from scipy.spatial.distance import cosine as cosinedist
from scipy.sparse import lil_matrix

# read in sentence similarity annotations
#sentenceDataFile = 'GS2011data-KSformat.txt'
sentenceDataFile = 'GS2013data-KSformat.txt' # like 2012 without the adjectives
#sentenceDataFile = 'GS2012data.txt'
#sentenceDataFile = 'KS2013-CoNLL.txt'
sentenceData = []
sentenceID = None
with open(sentenceDataFile, 'r') as fstr:
    annotatorScores = []
    for line in csv.DictReader(fstr, delimiter=' '):
        if sentenceID is None or line['sentence_id'] != sentenceID:
            if sentenceID is not None:
                sentenceData.append(thisSentence)
            sentenceID = line['sentence_id']
            thisSentence = {key: line[key] for key in line if key not in ['annotator_id', 'annotator_score']}
            thisSentence['similarity_scores'] = []
        thisSentence['similarity_scores'].append(float(line['annotator_score']))
    sentenceData.append(thisSentence)
print "found sentence annotations:", len(sentenceData)

# sort into pairs and find closest for each
lastSd = None
sentencePairs = []
for sd in sentenceData:
    if lastSd:
        matching = True
        for k in ['adj_subj', 'subj', 'landmark', 'adj_obj', 'obj']:
            if sd[k] != lastSd[k]:
                matching = False
        if matching:
            m0 = np.mean(lastSd['similarity_scores'])
            s0 = np.std(lastSd['similarity_scores'])
            m1 = np.mean(sd['similarity_scores'])
            s1 = np.std(sd['similarity_scores'])
            #print (m0,s0,m1,s1)
            # TODO: use std to check safety of categorisation
            cat = 0 if m0 > m1 else 1
            sentencePairs.append((lastSd,sd,cat))
            lastSd = None
        else:
            print "warning - no match found!", lastSd['sentence_id'], sd['sentence_id']
    else:
        lastSd = sd
print "paired sentence annotations", len(sentencePairs)
print sentencePairs[0]
print sentencePairs[1]

baseFolder = '/import/gijs-shared/gijs/spaces/'

W2V = 'W2V'
P300 = '300'
P2000 = '2000'

NOUN = '#NN'
VERB = '#VB'

vectorfiles = {}
tensorfiles = {}

vectorfiles[W2V] = 'word2vec_spaces/word2vecSpace.shelve'
tensorfiles[W2V] = 'word2vec_tensors/tensors_from_file_dims_300_combinedVerbs.txt_word2vecSpace.shelve'
#tensorfiles[W2V] = 'word2vec_tensors/tensors_from_file_gs2011ks2014verbs.txt_word2vecSpace.shelve'
#tensorfiles[W2V] = 'word2vec_tensors/tensors_from_file_ks2013verbs.txt_word2vecSpace.shelve'

vectorfiles[P300] = 'vector_spaces/vspace_gijs_raw_CW=5_DIMS=10000_mp_NORM_ppmi_CUTDOWN_2000.shelve'
tensorfiles[P300] = 'tensor_spaces/tensors_from_file_ks2013verbs.txt_vspace_gijs_raw_CW=5_DIMS=10000_mp_NORM_ppmi.shelve'

vectorfiles[P2000] = vectorfiles[P300]
tensorfiles[P2000] = 'tensor_spaces/tensors_dims_2000_from_file_ks2013verbs.txt_vspace_gijs_raw_CW=5_DIMS=10000_mp_NORM_ppmi.shelve'

for key in vectorfiles:
    vectorfiles[key] = os.path.join(baseFolder, vectorfiles[key])
    tensorfiles[key] = os.path.join(baseFolder, tensorfiles[key])


def loadSpaces(key):
    vectors = shelve.open(vectorfiles[key], 'r')
    tensors = shelve.open(tensorfiles[key], 'r')
    return vectors, tensors


def getVector(space, key, suffix, dims):
    vector = space.get(key+suffix)
    if vector is None:
        vector = space.get(key)
    if vector is None:
        print "vector not found:", key+suffix
        # US spelling in W2V
        newkey = None
        if re.search(r'is(e|es|ing|ed)\b', key) and suffix == VERB:
            newkey = re.sub(r'is(e|es|ing|ed)\b', r'iz\1', key)
        if re.search(r'offence\b', key):
            newkey = re.sub(r'offence\b', 'offense', key)
        if re.search(r'favour\b', key):
            newkey = re.sub(r'favour\b', 'favor', key)
        if re.search(r'-', key):
            newkey = re.sub(r'(\w+)-.*', r'\1', key)
        if newkey:
            print "try", newkey
            return getVector(space, newkey, suffix, dims)
    #return vector.toarray().flatten()[:dims]
    return vector[:dims]


def getSVOvectors(vSpace, tSpace, subjKey, verbKey, objKey):
    verbT = lil_matrix(tSpace.get(verbKey))
    dims = verbT.shape[0]
    subjV = getVector(vSpace, subjKey, NOUN, dims)
    objV = getVector(vSpace, objKey, NOUN, dims)
    #verbT = getVector(tSpace, verbKey, dims)
    return subjV, verbT, objV


def interpretSentence(S, V, O):
    res = {}
    # Grefenstette & Sadrzadeh 2011: (SxO) * V -> NxN matrix
    res['gs'] = (S.T.dot(O)).multiply(V)
    # Kartsaklis et al 2012 copy-subj: S * (VxO) -> Nx1 vector
    res['ks'] = S.T.multiply(V.dot(O.T))
    # Kartsaklis et al 2012 copy-subj: (SxV) * O = O * (V'xS) -> Nx1 vector
    res['ko'] = O.T.multiply(V.T.dot(S.T))
    return res


def doSentence(vSpace, tSpace, Skey, Vkey, Okey, Skeys, Vkeys, Okeys):
    S, V, O = getSVOvectors(vSpace, tSpace, Skey, Vkey, Okey)
    # identities for incremental representations
    OI = lil_matrix(np.ones(S.shape))
    VI = lil_matrix(np.eye(S.shape[1]))
    # tuples of possible continuations for incremental representations
    Os = [getVector(vSpace, objKey, NOUN, S.shape[0]) for objKey in Okeys]
    Vs = [lil_matrix(tSpace.get(verbKey)) for verbKey in Vkeys]
    # incremental results
    res = {}
    # using identity as the unseen information
    res['identity'] = [interpretSentence(S, VI, OI),
                       interpretSentence(S, V, OI),
                       interpretSentence(S, V, O)]
    # using V/O sum over possible completions as the unseen information
    res['sum'] = [interpretSentence(S, np.sum(Vs), np.sum(Os)),
                  interpretSentence(S, V, np.sum(Os)),
                  interpretSentence(S, V, O)]
    # using V/O direct sum over possible completions as the unseen information
    res1={}
    val=[interpretSentence(S, iV, iO) for iV in Vs for iO in Os]
    for cm in val[0].keys():
        res1[cm] = np.sum([v[cm] for v in val])
    res2={}
    val=[interpretSentence(S, V, iO) for iO in Os]
    for cm in val[0].keys():
        res2[cm] = np.sum([v[cm] for v in val])
    res['directsum'] = [res1,
                        res2,
                        interpretSentence(S, V, O)]
    return res

def hackMorphology(verb, suffix):
    if suffix == 'd':
        stem = re.sub(r'depositt$', r'deposit',
                      re.sub(r'([aeiou](t|l|p))$', r'\1\2', verb))
    else:
        stem = verb
    stem = re.sub(r'(ch|s|sh|x|z)$', r'\1e',
                  re.sub(r'(y)$', r'ie', stem))
    if suffix == 'd':
        stem = re.sub(r'([^aeiou])$', r'\1e', stem)
    return stem+suffix


def doBaseline(vSpace, tSpace, Skey, Vkey, Okey):
    dims = lil_matrix(tSpace.get(Vkey)).shape[0]
    S = getVector(vSpace, Skey, NOUN, dims)
    V = getVector(vSpace, Vkey, VERB, dims)
    # try:
    #     V = getVector(vSpace, hackMorphology(Vkey, 's'), VERB, dims)
    # except TypeError:
    #     print "giving up on", hackMorphology(Vkey, 's'), "try", Vkey
    #     V = getVector(vSpace, Vkey, VERB, dims)
    O = getVector(vSpace, Okey, NOUN, dims)
    # incremental results
    res = [S,
           S+V,
           S+V+O]
    return res

COMP_METHODS = ['gs', 'ks', 'ko', 'bs']
INCR_METHODS = ['identity', 'sum', 'directsum']

def testSentences(key):
    vSpace, tSpace = loadSpaces(key)
    # TODO incremental
    (tot, tot_y, tot_n) = ({}, {}, {})
    for im in INCR_METHODS:
        (tot[im], tot_y[im], tot_n[im]) = ({}, {}, {})
        for cm in COMP_METHODS:
            (tot[im][cm], tot_y[im][cm], tot_n[im][cm]) = ([0.0]*3, [0.0]*3, [0.0]*3)
    for s in sentencePairs:
        # sent is same for both elements in pair; only para(phrase) is different
        ss = set([v[i]['subj'] for i in [0,1] for v in sentencePairs if v[0]['landmark'] == s[0]['landmark']])
        vs = set([v[i]['verb'] for i in [0,1] for v in sentencePairs if v[0]['landmark'] == s[0]['landmark']])
        os = set([v[i]['obj'] for i in [0,1] for v in sentencePairs if v[0]['landmark'] == s[0]['landmark']])
        sent = doSentence(vSpace, tSpace, s[0]['subj'], s[0]['landmark'], s[0]['obj'], ss, vs, os)
        para0 = doSentence(vSpace, tSpace, s[0]['subj'], s[0]['verb'], s[0]['obj'], ss, vs, os)
        para1 = doSentence(vSpace, tSpace, s[1]['subj'], s[1]['verb'], s[1]['obj'], ss, vs, os)
        bs_sent = doBaseline(vSpace, tSpace, s[0]['subj'], s[0]['landmark'], s[0]['obj'])
        bs_para0 = doBaseline(vSpace, tSpace, s[0]['subj'], s[0]['verb'], s[0]['obj'])
        bs_para1 = doBaseline(vSpace, tSpace, s[1]['subj'], s[1]['verb'], s[1]['obj'])
        gold_cat = s[2]
        # incremental: test word-by-word
        for i in range(len(sent)):
            i2 = i # to compare against incremental paraphrase composition
            #i2 = len(sent)-1 # to compare against fullly composed paraphrase
            for im in tot.keys():
                for cm in tot[im].keys():
                    if cm == 'bs':
                        d0 = cosinedist( bs_sent[i].toarray().flatten(), bs_para0[i2].toarray().flatten() )
                        d1 = cosinedist( bs_sent[i].toarray().flatten(), bs_para1[i2].toarray().flatten() )
                    else:
                        d0 = cosinedist( sent[im][i][cm].toarray().flatten(), para0[im][i2][cm].toarray().flatten() )
                        d1 = cosinedist( sent[im][i][cm].toarray().flatten(), para1[im][i2][cm].toarray().flatten() )
                    if d0<d1:
                        lead_cat = 0
                    elif d1<d0:
                        lead_cat = 1
                    else:
                        #lead_cat = random.randint(0,1)
                        lead_cat = -1
                    if lead_cat < -0.5:
                        tot_y[im][cm][i] += 0.5
                        tot_n[im][cm][i] += 0.5
                    else:
                        if gold_cat == lead_cat:
                            tot_y[im][cm][i] += 1.0
                        else:
                            tot_n[im][cm][i] += 1.0
                    tot[im][cm][i] += 1.0
    for im in tot.keys():
        for cm in tot[im].keys():
            for i in range(3):
                print im, cm, i, tot[im][cm][i], tot_y[im][cm][i], tot_n[im][cm][i]
                print im, cm, i, "accuracy = ", tot_y[im][cm][i]/tot[im][cm][i]

if __name__ == '__main__':
    #testSentences('300')
    #testSentences('2000')
    testSentences('W2V')
