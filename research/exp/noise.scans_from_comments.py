import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
# nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *
# print(stopwords.words("english"))
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import unicodedata
from operator import itemgetter

def comments_to_words( raw_comment ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_comment,"html.parser").get_text() 
    # 2. Remove non-letters, non-numbers
    letters_only = re.sub("[^a-zA-Z0-9]", " ", review_text) 
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    # 3a.Stem the words so plurals are the same word
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    # 6. Join the words back into one string separated by space and return the result.
    return(letters_only, " ".join( meaningful_words ))


def get_comments_dirs(f):
    train = pd.read_csv(f,header=0,delimiter="~",quoting=3)
    # print(train.shape)
    # print(train.columns.values)
    paths=[]
    comms=[]
    raw=[]
    grp = train.groupby('"Path"')['"ScanComment"'].apply(lambda x: "{%s}" % ', '.join(x))
    for k in grp.keys():
        paths.append(k)
        (letters,words) = comments_to_words(grp[k])
        comms.append(words)
        raw.append(letters)
    # print(grouped.keys())
    # print(grouped)
    return [paths,comms,raw]

def comb_feat(feats_in,comp_list,succ):
    # manual hack to combine keywords that refer to the same thing, i.e. flair and flr
    feats_out=[]
    feats_out.append(sum(feats_in[[3,4]])) # flair: flair, flr
    feats_out.append(sum(feats_in[[1,2,24]])) # t1c: contrasst, contrast, t1c
    feats_out.append(sum(feats_in[[23,26,27]])) # t1p: t1, t1p, t1w,
    feats_out.append(sum(feats_in[[6,7,8,9,10]])) # gd: gad, gadolinium, gadolonium, gd, inject
    feats_out.append(sum(feats_in[[25]])) # t1g
    feats_out.append(sum(feats_in[[16,28,29,30]])) # t2w: newt2, t2, t2vol, t2w
    feats_out.append(sum(feats_in[[17,18,19,20]])) # pdw: pd, pdf, pdw, ppd # exclude pdf
    feats_out.append(sum(feats_in[[11,12,13,14,15]])) # mt: mt, mt0on, mtoff, mton, mtr
    # feats_out.append(sum(feats_in[[1,2]])) # contrast: contrasst, contrast
    # untracked: axt2flair, fspgr, psgr, rsfmri
    feats_out.append(int(sum(feats_out)>0))
    return feats_out

# We obtained a csv file with comments but each comment had newlines '\n' that our script does not know how to deal with it.  
# Therefore we needed to do the following to get the input csv named failed.csv from the provided csv named failedScans.csv
# >>>sed s/\"\,\"/\"~\"/g <failedScans.csv | sed ':a;N;$!ba;s/\"\n\"/\"***\"/g' | sed ':a;N;$!ba;s/\n//g' | tr '***' '\n' | sed  '/^$/d' >failed.csv
# vim failed.csv
# :%s/^M//g
csv_file = '/home/rpizarro/noise/sheets/failed.csv'
[paths,comms,raw] = get_comments_dirs(csv_file)
# print(comms[0:10])

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(comms)

# Numpy arrays are easy to work with, so convert the result to an array
train_data_features = train_data_features.toarray()

# print(train_data_features[0])
# print(sum(train_data_features[0]))

# print(train_data_features.shape)

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
# print(vocab)

# >>>include the token "pre" and "post"

acq = ['aftifact','align','angl','aqc','artefact','articact','artifact','axial','biogen','coil','coron','coronar','correct','coverag','delay','distort','dosag','dose','flip','fov','freq','frequenc','head','map','metal','motion','movement','ms','part','partial','qc','qualiti','rf','rfov','saggit','sagit','sagitt','sagittali','subject','tissu','tr','wraparound']
scans = ['axt2flair','contrasst','contrast','flair','flr','fspgr','gad','gadolinium','gadolonium','gd','inject','mt','mt0on','mtoff','mton','mtr','newt2','pd','pdf','pdw','ppd','psgr','rsfmri','t1','t1c','t1g','t1p','t1w','t2','t2vol','t2w']

# scans_succ = ['flr','t1c','t1p','gd','t1g','t2w','pdw','mt']
geometric = sorted(['distort','isocent','geometr','siena'])
intensity_p = sorted(['intens','bia','inhomogen','shim','artefact','metal','dental','brace','artifact','band','miss','slice','nois','signal','low','snr'])
intensity_m = sorted(['motion','ghost','brain']) # have to figure out how to incorporate this into the decision
movement = sorted(['motion','emot','movement'])
coverage_p = sorted(['coverag','ghost','top','brain','cut','cerebellum','wraparound','fov','rfov','posit','align'])
coverage_m = sorted(['motion'])
acquisition = sorted(['flow','tr','te','coil','paramet','encod','direct','ap','rl','phase','weight','acquisit','acquir','protocol','clinic'])
contrast = sorted(['gd','gad','gadolinium','gadolonium','contrast','enhanc','post','pre','inject','agent','dose'])


scans_idx_geom = [i for i,v in enumerate(vocab) for t in geometric if v==t]
scans_idx_intp = [i for i,v in enumerate(vocab) for t in intensity_p if v==t]
scans_idx_intm = [i for i,v in enumerate(vocab) for t in intensity_m if v==t]
scans_idx_move = [i for i,v in enumerate(vocab) for t in movement if v==t]
scans_idx_covp = [i for i,v in enumerate(vocab) for t in coverage_p if v==t]
scans_idx_covm = [i for i,v in enumerate(vocab) for t in coverage_m if v==t]
scans_idx_acqu = [i for i,v in enumerate(vocab) for t in acquisition if v==t]
scans_idx_cont = [i for i,v in enumerate(vocab) for t in contrast if v==t]

# print(scans_idx)
# out_sheet = [paths,raw,train_data_features[:,scans_idx]]
# cases_w_word = np.sum(train_data_features[:,scans_idx],axis=1)
# cases_idx = [i for i,c in enumerate(cases_w_word) if c>0]
# print(cases_idx)
# print(out_sheet[0][3],out_sheet[1][3])
# print(scans)
# print(out_sheet[2][3])

# out_sheet=[['Path','ScanComment'] + scans]
out_sheet=[]
total = np.zeros((7))
for i,p in enumerate(paths):
    # feats=comb_feat(train_data_features[i,scans_idx],scans,scans_succ)
    feats_geom = list(train_data_features[i,scans_idx_geom])
    feats_intp = list(train_data_features[i,scans_idx_intp])
    feats_intm = list(train_data_features[i,scans_idx_intm])
    feats_move = list(train_data_features[i,scans_idx_move])
    feats_covp = list(train_data_features[i,scans_idx_covp])
    feats_covm = list(train_data_features[i,scans_idx_covm])
    feats_acqu = list(train_data_features[i,scans_idx_acqu])
    feats_cont = list(train_data_features[i,scans_idx_cont])
    # print(len(feats_move))
    # print(len(feats_covp),len(feats_covm))
    feats = [] # feats_geom + feats_intp + feats_intm + feats_move + feats_covp + feats_covm 
    # print(len(feats))
    feats.append(int(sum(feats_geom)>0))
    if sum(feats_intm)>0:
        feats.append(0)
    else:
        feats.append(int(sum(feats_intp)>0))
    feats.append(int(sum(feats_move)>0))
    if sum(feats_covm)>0:
        feats.append(0)
    else:
        feats.append(int(sum(feats_covp)>0))

    if sum(feats)>0:
        feats.extend([0,0])
    elif sum(feats_cont)>0:
        feats.extend([1,0])
    elif sum(feats_acqu)>0:
        feats.extend([0,1])
    else:
        feats.extend([0,0])

    if sum(feats)>0:
        feats.extend([0])
    else:
        feats.extend([1])

    total = total + np.asarray(feats,dtype=int)
    out_sheet.append( [p.strip('"')] + feats + [ unicodedata.normalize('NFKD', raw[i]).encode('ascii','ignore') ] )


out_sheet = sorted(out_sheet, key=itemgetter(7),reverse=True)
out_sheet = sorted(out_sheet, key=itemgetter(6),reverse=True)
out_sheet = sorted(out_sheet, key=itemgetter(5),reverse=True)
out_sheet = sorted(out_sheet, key=itemgetter(4),reverse=True)
out_sheet = sorted(out_sheet, key=itemgetter(3),reverse=True)
out_sheet = sorted(out_sheet, key=itemgetter(2),reverse=True)
out_sheet = sorted(out_sheet, key=itemgetter(1),reverse=True)

out_sheet.append( ['Total'] + total.astype(int).tolist() + [''] )


# print(out_sheet[0:10])

# Add a column to signify if any of the scans are mentioned in the comments.  


file_name = '/home/rpizarro/noise/sheets/failed.geometric.csv'
# df = pd.DataFrame(out_sheet, columns=['Path'] + geometric + intensity_p + intensity_m + movement + coverage_p + coverage_m + ['Geometric','Intensity','Movement','Coverage','ScanComment'])
df = pd.DataFrame(out_sheet, columns=['Path'] + ['Geometric','Intensity','Movement','Coverage','Acquisition','Contrast','Other','ScanComment'])
df.to_csv(file_name,index=False)

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

[d,v]=[list(x) for x in zip(*sorted(zip(dist, vocab), key=itemgetter(0)))]

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
# for tag, count in zip(v, d):
    print(count,tag)

