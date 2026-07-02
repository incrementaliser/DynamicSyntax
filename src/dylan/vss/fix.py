"""
Data format conversion script for Purver, Sadrzadeh, Kempson, Wijnholds, Hough "Incremental Composition in Distributional Semantics"
Copyright Matthew Purver, QMUL 2019
"""

import csv, os

#inputFile = 'GS2011data.txt'
#outputFile = 'GS2011data-KSformat.txt'
inputFile = 'GS2013data.txt'
outputFile = 'GS2013data-KSformat.txt'

datas = []
sentences = {}
with open(inputFile, 'r') as fstr:
    for line in csv.DictReader(fstr, delimiter=' '):
        data = {}
        sent = line['verb'] + line['subject'] + line['object'] + line['landmark']
        if sent in sentences:
            sid = sentences[sent]
        else:
            sid = len(sentences) + 1
            sentences[sent] = sid
        data['sentence_id'] = sid
        data['annotator_id'] = line['participant'].replace('participant','')
        data['adj_subj'] = 'dummy'
        data['adj_obj'] = 'dummy'
        data['subj'] = line['subject']
        data['obj'] = line['object']
        data['landmark'] = line['verb'] # GS2011 uses different view: landmark changes, verb stays same
        data['verb'] = line['landmark']
        data['annotator_score'] = line['input']
        datas.append(data)
print "found sentence annotations:", len(datas)

# used for gs2011
#datas = sorted(sorted(sorted(datas, key=lambda x: int(x['annotator_id'])), key=lambda x: int(x['sentence_id'])), key=lambda x: x['subj'])
# used for gs2013
datas = sorted(sorted(sorted(sorted(datas, key=lambda x: int(x['annotator_id'])), key=lambda x: int(x['sentence_id'])), key=lambda x: x['subj']), key=lambda x: x['obj'])

with open(outputFile, 'w') as fstr:
    wr = csv.DictWriter(fstr, ['sentence_id', 'annotator_id', 'adj_subj', 'subj', 'landmark', 'verb', 'adj_obj', 'obj', 'annotator_score'], delimiter=' ')
    wr.writeheader()
    for data in datas:
        wr.writerow(data)
