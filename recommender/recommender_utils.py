import os
import re
import sys
import time
from datetime import datetime
import random as rndm
from itertools import groupby
from config import config
import urllib
import requests
from numpy import *
import operator
import cPickle
import pymongo

# Helper functions
def req(url, **kwargs):
    """
    Function to query Solr
    """
    kwargs['wt'] = 'json'
    query_params = urllib.urlencode(kwargs)
    r = requests.get(url, params=query_params)
    return r.json()

def flatten(items):
    """flatten(sequence) -> list

    Returns a single, flat list which contains all elements retrieved
    from the sequence and all recursively contained sub-sequences
    (iterables).

    Examples:
    >>> [1, 2, [3,4], (5,6)]
    [1, 2, [3, 4], (5, 6)]
    >>> flatten([[[1,2,3], (42,None)], [4,5], [6], 7, MyVector(8, 9, 10)])
    [1, 2, 3, 42, None, 4, 5, 6, 7, 8, 9, 10]"""

    result = []
    for item in items:
        if hasattr(item, '__iter__'):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def multikeysort(items, columns):
# When we switch over to a later Python version than 2.4, we can use the
# the list comprehension below (Python 2.4 does not support conditions)
#    comparers = [ ((operator.itemgetter(col[1:].strip()), -1) if col.startswith('-') else (operator.itemgetter(col.strip()), 1)) for col in columns]
    comparers = []
    for col in columns:
        if col.startswith('-'):
            comparers.append((operator.itemgetter(col[1:].strip()), -1))
        else:
            comparers.append((operator.itemgetter(col.strip()), 1))

    def comparer(left, right):
        for fn, mult in comparers:
            result = cmp(fn(left), fn(right))
            if result:
                return mult * result
        else:
            return 0
    return sorted(items, cmp=comparer)

def get_before_after(item,list):
    idx = list.index(item)
    try:
        before = list[idx-1]
    except:
        before = "NA"
    try:
        after = list[idx+1]
    except:
        after = "NA"
    return [before,after]

def get_frequencies(l):
    tmp = [(k,len(list(g))) for k, g in groupby(sorted(l))]
    return sorted(tmp, key=operator.itemgetter(1),reverse=True)[:100]

def make_date(datestring):
    pubdate = map(lambda a: int(a), datestring.split('-'))
    if pubdate[1] == 0:
        pubdate[1] = 1
    return datetime(pubdate[0],pubdate[1],1)

def uniq(items):
    """
    Returns a uniqued list.
    """
    unique = []
    unique_dict = {}
    for item in items:
        if item not in unique_dict:
            unique_dict[item] = None
            unique.append(item)
    return unique
# Helper Functions: Data Retrieval
def get_normalized_keywords(bibc):
    keywords = []
    q = 'bibcode:%s or references(bibcode:%s)' % (bibc,bibc)
    fl= 'keyword_norm'
    rsp = req(config.SOLR_URL, q=q, fl=fl, rows=config.MAX_HITS)
    for doc in rsp['response']['docs']:
        try:
            keywords += doc['keyword_norm']
        except:
            pass
    return filter(lambda a: a in config.IDENTIFIERS, keywords)

def get_article_data(biblist):
    fl = 'bibcode,reference,citation_count,pubdate'
    sort='pubdate desc'
    list = " OR ".join(map(lambda a: "bibcode:%s"%a, biblist))
    q = '%s' % list
    rsp = req(config.SOLR_URL, q=q, sort=sort, fl=fl, rows=config.MAX_HITS)
    results = rsp['response']['docs']
    results = filter(lambda a: 'reference' in a, results) 
    return results

def get_citations(biblist):
    citations = []
    fl = 'bibcode,reference'
    list = " OR ".join(map(lambda a: "bibcode:%s"%a, biblist))
    q = 'citations(%s)' % list
    rsp = req(config.SOLR_URL, q=q, fl=fl, rows=config.MAX_HITS)
    results = rsp['response']['docs']
    results = filter(lambda a: 'reference' in a, results)
    for doc in results:
        multi = len(filter(lambda a: a in biblist, doc['reference']))
        citations += [doc['bibcode']]*multi
    return citations

def get_projection_matrix(cluster):
    data_file = config.CLUSTER_PROJECTION_FILE % cluster
    try:
        data = open(data_file).read().strip().split('\n')
    except:
        data = []
    matrix = []
    for entry in data:
        matrix.append(map(lambda a: float(a),entry.split()))
    return array(matrix)

def get_recently_viewed(cookie):
    bibcodes = []
    if len(cookie) != 10:
        return bibcodes
    cookie = cookie.strip()
    lines = urllib.urlopen(config.RECENTS_URL%cookie).read().strip().split('\n')
    bibrecs  = filter(lambda a: a.strip()[:9] == '<bibcode>',lines)
    bibcodes = map(lambda a: re.sub('</?bibcode>','',a).replace('&amp;','&'), bibrecs)
    return bibcodes
#   
# Helper Functions: Data Processing
def make_paper_vector(bibc):
    data = get_normalized_keywords(bibc)
    if len(data) == 0:
        return []
    freq = dict((config.IDENTIFIERS.index(x), float(data.count(x))/float(len(data))) for x in data)
    FreqVec = [0.0]*len(config.IDENTIFIERS)
    for i in freq.keys():
        FreqVec[i] = freq[i]
    return FreqVec

def project_paper(pvector,pcluster=None):
    if not pcluster:
        pcluster = -1
    client = pymongo.MongoClient(config.MONGO_HOST,config.MONGO_PORT)
    db = client.recommender
    db.authenticate(config.MONGO_USER,config.MONGO_PWD)
    collection = db.matrices
    res = collection.find_one({'cluster':int(pcluster)})
    PROJECTION = cPickle.loads(res['projection_matrix'])
    PaperVector = array(pvector)
    try:
        coords = dot(PaperVector,PROJECTION)
    except:
        coords = []
    return coords

def find_paper_cluster(pvec,bibc):
    client = pymongo.MongoClient(config.MONGO_HOST,config.MONGO_PORT)
    db = client.recommender
    db.authenticate(config.MONGO_USER,config.MONGO_PWD)
    collection = db.clusters
    res = collection.find_one({'members':bibc})
    if res:
        return res['cluster']

    min_dist = 9999
    clusters = collection.find()
    for entry in clusters:
        centroid = cPickle.loads(entry['centroid'])
        dist = linalg.norm(pvec-array(centroid))
        if dist < min_dist:
            cluster = entry['cluster']
        min_dist = min(dist, min_dist)
    return str(cluster)

def find_cluster_papers(pcluster):
    result = []
    client = pymongo.MongoClient(config.MONGO_HOST,config.MONGO_PORT)
    db = client.recommender
    db.authenticate(config.MONGO_USER,config.MONGO_PWD)
    cluster_coll = db.recent_paper_clustering
    entries = cluster_coll.find({'cluster':int(pcluster)})
    for entry in entries:
        result.append(entry)
    return result

def find_closest_cluster_papers(pcluster,vec):
    client = pymongo.MongoClient(config.MONGO_HOST,config.MONGO_PORT)
    db = client.recommender
    db.authenticate(config.MONGO_USER,config.MONGO_PWD)
    cluster_coll = db.clusters
    paper_coll = client.recommender.clustering
    res = cluster_coll.find_one({'cluster':int(pcluster)})
    distances = []
    for paper in res['members']:
        res = paper_coll.find_one({'paper':paper})
        if res:
            cvector = cPickle.loads(res['vector_low'])
        else:
            continue
        dist = linalg.norm(vec-cvector)
        distances.append((paper,dist))
    d = sorted(distances, key=operator.itemgetter(1),reverse=False)
    return map(lambda a: a[0],d[:config.MAX_NEIGHBORS])

def get_recommendations(G,remove=None):
    client = pymongo.MongoClient(config.MONGO_HOST,config.MONGO_PORT)
    db = client.recommender
    db.authenticate(config.MONGO_USER,config.MONGO_PWD)
    reads_coll = db.reads
    # get all reads series by frequent readers who read
    # any of the closest papers (stored in G)
    res = reads_coll.find({'reads':{'$in':G}})
    # lists to record papers read just before and after a paper
    # was read from those closest papers, and those to calculate
    # associated frequency distributions
    before = []
    BeforeFreq = []
    after  = []
    AfterFreq = []
    # list to record papers read by people who read one of the
    # closest papers
    alsoreads = []
    AlsoFreq = []
    # start processing those reads we determined earlier
    for item in res:
        alsoreads += item['reads']
        overlap = list(set(item['reads']) & set(G))
        before_after_reads = map(lambda a: get_before_after(a, item['reads']), overlap)
        for reads_pair in before_after_reads:
            before.append(reads_pair[0])
            after.append(reads_pair[1])
    # remove all "NA"
    before = filter(lambda a: a != "NA", before)
    after  = filter(lambda a: a != "NA", after)
    # remove (if specified) the paper for which we get recommendations
    if remove:
        alsoreads = filter(lambda a: a != remove, alsoreads)
    # calculate frequency distributions
    BeforeFreq = get_frequencies(before)
    AfterFreq  = get_frequencies(after)
    AlsoFreq  = get_frequencies(alsoreads)
    # get publication data for the top 100 most alsoread papers
    top100 = map(lambda a: a[0], AlsoFreq)
    top100_data = get_article_data(top100)
    mostRecent = top100_data[0]['bibcode']
    top100_data = sorted(top100_data, key=operator.itemgetter('citation_count'),reverse=True)
    # get the most cited paper from the top 100 most alsoread papers
    MostCited = top100_data[0]['bibcode']
    # get the most papers cited BY the top 100 most alsoread papers
    # sorted by citation
    refs100 = flatten(map(lambda a: a['reference'], top100_data))
    RefFreq = get_frequencies(refs100)
    # get the papers that cite the top 100 most alsoread papers
    # sorted by frequency
    cits100 = get_citations(top100)
    CitFreq = get_frequencies(cits100)
    # now we have everything to build the recommendations
    FieldNames = 'Field definitions:'
    Recommendations = []
    Recommendations.append(FieldNames)
    Recommendations.append(G[0])
    Recommendations.append(BeforeFreq[0][0])
    if AfterFreq[0][0] == BeforeFreq[0][0]:
        try:
            Recommendations.append(AfterFreq[1][0])
        except:
            Recommendations.append(AfterFreq[0][0])
    else:
        Recommendations.append(AfterFreq[0][0])
    try:
        Recommendations.append(rndm.choice(AlsoFreq[:10])[0])
    except:
        Recommendations.append(AlsoFreq[0][0])
    Recommendations.append(mostRecent)
    try:
        Recommendations.append(rndm.choice(CitFreq[:10])[0])
    except:
        Recommendations.append(CitFreq[0][0])
    try:
        Recommendations.append(rndm.choice(RefFreq[:10])[0])
    except:
        Recommendations.append(RefFreq[0][0])
    Recommendations.append(MostCited)
    RecommDict = {}
    RecommDict['Closest']      =Recommendations[1]
    RecommDict['ReadBefore']   =Recommendations[2]
    RecommDict['ReadAfter']    =Recommendations[3]
    RecommDict['MostAlsoRead'] =Recommendations[4]
    RecommDict['MostRecent100']=Recommendations[5]
    RecommDict['MostCited100'] =Recommendations[6]
    RecommDict['MostRefer100'] =Recommendations[7]
    RecommDict['MostCited']    =Recommendations[8]

    return Recommendations

def get_suggestions(**args):
    if 'bibcodes' in args:
        biblist = args['bibcodes']
    elif 'cookie' in args:
        biblist = get_recently_viewed(args['cookie'])
    suggestions = []
    Nselect = config.MAX_INPUT
    input_data = get_article_data(biblist)
    for entry in input_data[:config.INPUT_LIMIT]:
        bibcode = entry['bibcode']
        pdate_p = make_date(entry['pubdate'])
        vec = make_paper_vector(bibcode)
        papervec = array(vec)
        if len(vec) == 0:
            continue
        try:
            pvec = project_paper(vec)
            pclust=find_paper_cluster(pvec,bibcode)
        except:
            continue
        clusterPapers = find_cluster_papers(pclust)
        Nclusterpapers= len(clusterPapers)
        if Nclusterpapers == 0:
            continue
        if Nclusterpapers > Nselect:
            selection = rndm.sample(range(Nclusterpapers),Nselect)
            clusterSelection = map(lambda a: clusterPapers[a], selection)
        else:
            clusterSelection = clusterPapers
        resdicts = []
        for entry in clusterSelection:
            paper   = entry['paper']
            reads   = entry['reads']
            pdate_c = entry['pubdate']
            cpvec = cPickle.loads(entry['vector'])
            # remove potential suggestions that are in the
            # submitted list of bibcodes, or in already
            # generated suggestions
            if paper in biblist or paper in suggestions:
                continue
            # calculate reads rate
            time_span = max(1,abs((pdate_c - pdate_p).days))
            multiplier = float(reads)/float(time_span)
            dist = linalg.norm(papervec-cpvec)
            resdicts.append({'bibcode':paper, 'year':int(paper[:4]), 'distance':dist, 'reads':multiplier})
        # sort by year (high to low), then distance (low to high),
        # then reads (high to low)
        a = multikeysort(resdicts, ['-year', 'distance', '-reads'])
        suggestions.append(a[0]['bibcode'])
        if len(suggestions) == config.SUGGEST_NUMBER:
            break
    return suggestions
