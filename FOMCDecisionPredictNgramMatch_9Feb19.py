#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 15:58:23 2019

@author: nitinsinghal
"""

# Sentiment Analysis of FOMC statement using NLP NLTK
# Using tokenization, word count, sentence importance
from builtins import range
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

stopwords = set(w.rstrip() for w in open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/NLPCourse/machine_learning_examples-master/nlp_class/stopwords.txt'))

# add more stopwords to remove from the text
stopwords = stopwords.union({'federal', 'fund', 'staff', 'reported', 'committee', 'discussed', 'desk', 'dealer', 
                            'meeting', 'agency', 'mortgage-backed', 'security', 'minute', 'board', 'forecaster', 
                            'governor', 'president', 'mexican', 'north', 'technology', 'example', 'addendum', 
                            'corporation', 'interbank', 'technological', 'requirement', 'seasonality'})

# Load FOMC statement documents
fomcdec99text = open('/Users/FOMCDec99Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov99text = open('/Users/FOMCNov99Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcoct99text = open('/Users/FOMCOct99Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug99text = open('/Users/FOMCAug99Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun99text = open('/Users/FOMCJun99Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay99text = open('/Users/FOMCMay99Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar99text = open('/Users/FOMCMar99Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcfeb99text = open('/Users/FOMCFeb99Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec00text = open('/Users/FOMCDec00Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov00text = open('/Users/FOMCNov00Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcoct00text = open('/Users/FOMCOct00Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug00text = open('/Users/FOMCAug00Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun00text = open('/Users/FOMCJun00Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay00text = open('/Users/FOMCMay00Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar00text = open('/Users/FOMCMar00Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcfeb00text = open('/Users/FOMCFeb00Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec01text = open('/Users/FOMCDec01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov01text = open('/Users/FOMCNov01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcoct01text = open('/Users/FOMCOct01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep01text = open('/Users/FOMCSep01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug01text = open('/Users/FOMCAug01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun01text = open('/Users/FOMCJun01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay01text = open('/Users/FOMCMay01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcapr01text = open('/Users/FOMCApr01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar01text = open('/Users/FOMCMar01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomc31jan01text = open('/Users/FOMC31Jan01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomc03jan01text = open('/Users/FOMC03Jan01Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec02text = open('/Users/FOMCDec02Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov02text = open('/Users/FOMCNov02Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep02text = open('/Users/FOMCSep02Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug02text = open('/Users/FOMCAug02Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun02text = open('/Users/FOMCJun02Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay02text = open('/Users/FOMCMay02Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar02text = open('/Users/FOMCMar02Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan02text = open('/Users/FOMCJan02Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec03text = open('/Users/FOMCDec03Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcoct03text = open('/Users/FOMCOct03Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep03text = open('/Users/FOMCSep03Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug03text = open('/Users/FOMCAug03Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun03text = open('/Users/FOMCJun03Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay03text = open('/Users/FOMCMay03Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar03text = open('/Users/FOMCMar03Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan03text = open('/Users/FOMCJan03Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec04text = open('/Users/FOMCDec04Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov04text = open('/Users/FOMCNov04Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep04text = open('/Users/FOMCSep04Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug04text = open('/Users/FOMCAug04Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun04text = open('/Users/FOMCJun04Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay04text = open('/Users/FOMCMay04Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar04text = open('/Users/FOMCMar04Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan04text = open('/Users/FOMCJan04Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec05text = open('/Users/FOMCDec05Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov05text = open('/Users/FOMCNov05Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep05text = open('/Users/FOMCSep05Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug05text = open('/Users/FOMCAug05Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun05text = open('/Users/FOMCJun05Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay05text = open('/Users/FOMCMay05Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar05text = open('/Users/FOMCMar05Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcfeb05text = open('/Users/FOMCFeb05Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec06text = open('/Users/FOMCDec06Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcoct06text = open('/Users/FOMCOct06Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep06text = open('/Users/FOMCSep06Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug06text = open('/Users/FOMCAug06Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun06text = open('/Users/FOMCJun06Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay06text = open('/Users/FOMCMay06Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar06text = open('/Users/FOMCMar06Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan06text = open('/Users/FOMCJan06Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec07text = open('/Users/FOMCDec07Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcoct07text = open('/Users/FOMCOct07Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep07text = open('/Users/FOMCSep07Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug07text = open('/Users/FOMCAug07Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun07text = open('/Users/FOMCJun07Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay07text = open('/Users/FOMCMay07Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar07text = open('/Users/FOMCMar07Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan07text = open('/Users/FOMCJan07Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec08text = open('/Users/FOMCDec08Stmt.txt', encoding='utf-8', errors='ignore').read()
fomc8oct08text = open('/Users/FOMC8Oct08Stmt.txt', encoding='utf-8', errors='ignore').read()
fomc29oct08text = open('/Users/FOMC29Oct08Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep08text = open('/Users/FOMCSep08Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug08text = open('/Users/FOMCAug08Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun08text = open('/Users/FOMCJun08Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcapr08text = open('/Users/FOMCApr08Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar08text = open('/Users/FOMCMar08Stmt.txt', encoding='utf-8', errors='ignore').read()
fomc22jan08text = open('/Users/FOMC22Jan08Stmt.txt', encoding='utf-8', errors='ignore').read()
fomc30jan08text = open('/Users/FOMC30Jan08Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec09text = open('/Users/FOMCDec09Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov09text = open('/Users/FOMCNov09Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep09text = open('/Users/FOMCSep09Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug09text = open('/Users/FOMCAug09Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun09text = open('/Users/FOMCJun09Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcapr09text = open('/Users/FOMCApr09Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar09text = open('/Users/FOMCMar09Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan09text = open('/Users/FOMCJan09Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec14text = open('/Users/FOMCDec14Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcoct14text = open('/Users/FOMCOct14Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep14text = open('/Users/FOMCSep14Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjul14text = open('/Users/FOMCJul14Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun14text = open('/Users/FOMCJun14Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar14text = open('/Users/FOMCMar14Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcapr14text = open('/Users/FOMCApr14Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan14text = open('/Users/FOMCJan14Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec15text = open('/Users/FOMCDec15Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcoct15text = open('/Users/FOMCOct15Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep15text = open('/Users/FOMCSep15Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjul15text = open('/Users/FOMCJul15Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun15text = open('/Users/FOMCJun15Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar15text = open('/Users/FOMCMar15Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcapr15text = open('/Users/FOMCApr15Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan15text = open('/Users/FOMCJan15Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec16text = open('/Users/FOMCDec16Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov16text = open('/Users/FOMCNov16Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep16text = open('/Users/FOMCSep16Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjul16text = open('/Users/FOMCJul16Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun16text = open('/Users/FOMCJun16Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar16text = open('/Users/FOMCMar16Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcapr16text = open('/Users/FOMCApr16Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan16text = open('/Users/FOMCJan16Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec17text = open('/Users/FOMCDec17Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov17text = open('/Users/FOMCNov17Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep17text = open('/Users/FOMCSep17Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjul17text = open('/Users/FOMCJul17Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun17text = open('/Users/FOMCJun17Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay17text = open('/Users/FOMCMay17Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar17text = open('/Users/FOMCMar17Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcfeb17text = open('/Users/FOMCFeb17Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec18text = open('/Users/FOMCDec18Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov18text = open('/Users/FOMCNov18Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep18text = open('/Users/FOMCSep18Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug18text = open('/Users/FOMCAug18Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun18text = open('/Users/FOMCJun18Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay18text = open('/Users/FOMCMay18Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar18text = open('/Users/FOMCMar18Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan18text = open('/Users/FOMCJan18Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcjan19text = open('/Users/FOMCJan19Stmt.txt', encoding='utf-8', errors='ignore').read()

ratehikefiles = [fomcmar18text, fomcjun18text, fomcsep18text, fomcdec17text, fomcjun17text, fomcmar17text, 
                 fomcdec15text, fomcdec18text, fomcdec05text, fomcnov05text, fomcsep05text, fomcaug05text, 
                 fomcjun05text, fomcmay05text, fomcmar05text, fomcfeb05text, fomcjan06text, fomcmar06text, 
                 fomcmay06text, fomcjun06text, fomcjun04text, fomcaug04text, fomcsep04text, fomcnov04text, 
                 fomcdec04text, fomcjun99text, fomcaug99text, fomcnov99text, fomcfeb00text, fomcmar00text, 
                 fomcmay00text] #fomcdec16text
noratehikefiles = [fomcnov18text, fomcaug18text, fomcmay18text, fomcjan18text, fomcfeb17text, fomcjul17text, 
                   fomcmay17text, fomcnov17text, fomcsep17text, fomcnov16text, fomcsep16text, fomcjul16text, 
                   fomcjul15text, fomcmar16text, fomcapr16text, fomcjan16text, fomcoct15text, fomcsep15text, 
                   fomcoct14text, fomcjun15text, fomcmar15text, fomcapr15text, fomcjan15text, fomcdec14text, 
                   fomcjan14text, fomcjul14text, fomcjun14text, fomcmar14text, fomcapr14text, fomcsep14text, 
                   fomcdec09text, fomcnov09text, fomcsep09text, fomcaug09text, fomcjun09text, fomcjun16text, 
                   fomcapr09text, fomcmar09text, fomcjan09text, fomcjan04text, fomcmar04text, fomcmay04text, 
                   fomcdec03text, fomcoct03text, fomcsep03text, fomcaug03text, fomcmay03text, fomcmar03text,  
                   fomcsep08text, fomcaug08text, fomcjun08text, fomcaug07text, fomcjun07text, fomcmay07text, 
                   fomcmar07text, fomcjan07text, fomcdec06text, fomcoct06text, fomcsep06text, fomcaug06text, 
                   fomcdec00text, fomcnov00text, fomcoct00text, fomcaug00text, fomcjun00text, fomcjan03text, 
                   fomcdec99text, fomcoct99text, fomcmay99text, fomcmar99text, fomcfeb99text, fomcdec02text, 
                   fomcsep02text, fomcaug02text, fomcjun02text, fomcmay02text, fomcmar02text, fomcjan02text]
                    # 
ratecutfiles = [fomcsep07text, fomcoct07text, fomcdec07text, fomc30jan08text, fomc22jan08text, fomcmar08text, 
                fomc29oct08text, fomc8oct08text, fomcapr08text, fomcjun03text, fomcdec01text, fomcdec08text, 
                fomcnov01text, fomcoct01text, fomcaug01text, fomcjun01text, fomcmay01text, fomcapr01text, 
                fomcmar01text, fomc31jan01text, fomc03jan01text, fomcsep01text, fomcnov02text]
                #

fomcfiles = [ratehikefiles, noratehikefiles, ratecutfiles]
categories = ['hike', 'nohike', 'cut']

# As there are more noratehike, ratehike files compared to ratecut files, we will take random sample 
# of noratehike, ratehike = ratecut files
# This ensures that we are comparing equal no of files to create trigrams
np.random.shuffle(noratehikefiles)
noratehikefileseqtocut = noratehikefiles[:len(ratecutfiles)]
np.random.shuffle(ratehikefiles)
ratehikefileseqtocut = ratehikefiles[:len(ratecutfiles)]

# create a function to tokenize text. Remove short words less than 2 alphabets
# then convert to trigrams, fourgrams, fivegrams
def text_tokenize(texttoken):
    txt = texttoken
    txt = txt.lower()
    tokens = nltk.tokenize.word_tokenize(txt)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    return tokens

def token_trigram(fomctext): 
    tritokens = text_tokenize(fomctext)
    trigrams = {}    
    for i in range(len(tritokens) - 2):
        k = (tritokens[i], tritokens[i+1], tritokens[i+2])
        if k not in trigrams:
            trigrams[k] = 1
        else:
            trigrams[k] += 1
    return trigrams

#def token_fourgram(fomctext): 
#    fourtokens = text_tokenize(fomctext)
#    fourgrams = {}
#    for j in range(len(fourtokens) - 3):
#        l = (fourtokens[j], fourtokens[j+1], fourtokens[j+2], fourtokens[j+3])
#        if l not in fourgrams:
#            fourgrams[l] = 1
#        else:
#            fourgrams[l] += 1
#    return fourgrams
#
#def token_fivegram(fomctext): 
#    fivetokens = text_tokenize(fomctext)
#    fivegrams = {}
#    for m in range(len(fivetokens) - 4):
#        n = (fivetokens[m], fivetokens[m+1], fivetokens[m+2], fivetokens[m+3], fivetokens[m+4])
#        if n not in fivegrams:
#            fivegrams[n] = 1
#        else:
#            fivegrams[n] += 1
#    return fivegrams

# apply to each text file - convert to trigrams
# create a dictionary with trigram index & frequency for hike and no hike files
# keep trigram as key and frequency as value, then test on new document
hiketrigramdict = {}
nohiketrigramdict = {}
ratecuttrigramdict = {}

for hike in ratehikefileseqtocut:
    hiketext = hike
    hiketrigram = token_trigram(hiketext)
    for htgkey, htgval in hiketrigram.items():
        if htgkey not in hiketrigramdict:
            hiketrigramdict[htgkey] = htgval
        else:
            hiketrigramdict[htgkey] += htgval

for nohike in noratehikefileseqtocut:
    nohiketext = nohike
    nohiketrigram = token_trigram(nohiketext)
    for nhtgkey, nhtgval in nohiketrigram.items():
        if nhtgkey not in nohiketrigramdict:
            nohiketrigramdict[nhtgkey] = nhtgval
        else:
            nohiketrigramdict[nhtgkey] += nhtgval

for ratecut in ratecutfiles:
    ratecuttext = ratecut
    ratecuttrigram = token_trigram(ratecuttext)
    for rctgkey, rctgval in ratecuttrigram.items():
        if rctgkey not in ratecuttrigramdict:
            ratecuttrigramdict[rctgkey] = rctgval
        else:
            ratecuttrigramdict[rctgkey] += rctgval

# As there are more noratehike, ratehike trigrams compared to ratecut trigrams, we will take random sample 
# of noratehike, ratehike = ratecut trigrams by shuffling the keys of the dict
# This ensures that we are comparing equal no of trigrams 
nohikedictkeys = list(nohiketrigramdict.keys())
hikedictkeys = list(hiketrigramdict.keys())
np.random.shuffle(nohikedictkeys)
nohikedictkeys = nohikedictkeys[:len(ratecuttrigramdict)]
np.random.shuffle(hikedictkeys)
hikedictkeys = hikedictkeys[:len(ratecuttrigramdict)]
nohiketrigramdicteqtocut = {}
hiketrigramdicteqtocut = {}
for shuflnhkey in nohikedictkeys:
    nohiketrigramdicteqtocut[shuflnhkey] = nohiketrigramdict[shuflnhkey]
for shuflhkey in hikedictkeys:
    hiketrigramdicteqtocut[shuflhkey] = hiketrigramdict[shuflhkey]

# take a new fomc document and check hike/nohike counts
# basic idea for ngram count to see which category has higher occurence
# then predict which category (hike/nohike) it belongs to
predhiketrigramdict = {}
hikecount = 0
nohikecount = 0
ratecutcount = 0
nomatch = 0

#predict_text = fomcnov02text #ratecut
#predict_text = fomcjun16text #nohike
predict_text = fomcdec16text #hike
#predict_text = fomcjan19text

predict_trigram = token_trigram(predict_text)

# simple count for trigram match considering unique or common occurence across 3 file sets
for pkey, pval in predict_trigram.items():
    if pkey in hiketrigramdicteqtocut:
        if pkey not in nohiketrigramdicteqtocut:
            if pkey not in ratecuttrigramdict:
                hikecount += 1 # trigram only in hiketrigramdicteqtocut
            elif pkey in ratecuttrigramdict:
                if hiketrigramdicteqtocut[pkey] > ratecuttrigramdict[pkey]:
                    hikecount += 1 
                elif ratecuttrigramdict[pkey] > hiketrigramdicteqtocut[pkey]:
                    ratecutcount += 1 
        elif pkey in nohiketrigramdicteqtocut:
            if pkey not in ratecuttrigramdict:
                if hiketrigramdicteqtocut[pkey] > nohiketrigramdicteqtocut[pkey]:
                    hikecount += 1 
                elif nohiketrigramdicteqtocut[pkey] > hiketrigramdicteqtocut[pkey]:
                    nohikecount += 1
            elif pkey in ratecuttrigramdict:
                if (hiketrigramdicteqtocut[pkey] > nohiketrigramdicteqtocut[pkey]) and (hiketrigramdicteqtocut[pkey] > ratecuttrigramdict[pkey]):
                    hikecount += 1 
                elif (nohiketrigramdicteqtocut[pkey] > hiketrigramdicteqtocut[pkey]) and (nohiketrigramdicteqtocut[pkey] > ratecuttrigramdict[pkey]):
                    nohikecount += 1
                elif (ratecuttrigramdict[pkey] > hiketrigramdicteqtocut[pkey]) and (ratecuttrigramdict[pkey] > nohiketrigramdicteqtocut[pkey]):
                    ratecutcount += 1
    elif pkey in nohiketrigramdicteqtocut:
        if pkey not in ratecuttrigramdict:
                nohikecount += 1 # trigram only in hiketrigramdict
        elif pkey in ratecuttrigramdict:
                if nohiketrigramdicteqtocut[pkey] > ratecuttrigramdict[pkey]:
                    nohikecount += 1 
                elif ratecuttrigramdict[pkey] > nohiketrigramdicteqtocut[pkey]:
                    ratecutcount += 1 
    elif pkey in ratecuttrigramdict:
        ratecutcount += 1 # trigram only in ratecuttrigramdict
    else:
        nomatch += 1

hikefrac = hikecount/(hikecount+nohikecount+ratecutcount)
nohikefrac = nohikecount/(hikecount+nohikecount+ratecutcount)
ratecufrac = nohikecount/(hikecount+nohikecount+ratecutcount)
hikefracfull = hikecount/(hikecount+nohikecount+ratecutcount+nomatch)
nohikefracfull = nohikecount/(hikecount+nohikecount+ratecutcount+nomatch)
ratecutfrac = ratecutcount/(hikecount+nohikecount+ratecutcount+nomatch)
nomatchfrac = nomatch/(hikecount+nohikecount+ratecutcount+nomatch)

print(hikecount, ":hikecount, ", nohikecount, ":nohikecount, ", ", ", ratecutcount, ":ratecutcount", ", ", 
      nomatch, ":nomatch", ", ")
print(hikefrac, ":hikefrac, ", nohikefrac, ":nohikefrac, ", ", ")
print(hikefracfull, ":hikefracfull, ", nohikefracfull, ":nohikefracfull, ",  ratecutfrac, ":ratecutfrac, ", 
      nomatchfrac, ":nomatchfrac")
if hikecount > nohikecount:
    print(": Rate Hike")
else:
    print(": No Rate Hike")

# fractional count for frequency for trigram match considering unique or common occurence across 3 file sets
for pkey, pval in predict_trigram.items():
    if pkey in hiketrigramdicteqtocut:
        if pkey not in nohiketrigramdicteqtocut:
            if pkey not in ratecuttrigramdict:
                hikecount += 1 # trigram only in hiketrigramdicteqtocut
            elif pkey in ratecuttrigramdict:
                hikecount += pval/(pval+ratecuttrigramdict[pkey])
                ratecutcount += ratecuttrigramdict[pkey]/(pval+ratecuttrigramdict[pkey])
        elif pkey in nohiketrigramdicteqtocut:
            if pkey not in ratecuttrigramdict:
                hikecount += pval/(pval+nohiketrigramdicteqtocut[pkey])
                nohikecount += nohiketrigramdicteqtocut[pkey]/(pval+nohiketrigramdicteqtocut[pkey])
            elif pkey in ratecuttrigramdict:
                hikecount += pval/(pval+nohiketrigramdicteqtocut[pkey]+ratecuttrigramdict[pkey])
                nohikecount += nohiketrigramdicteqtocut[pkey]/(pval+nohiketrigramdicteqtocut[pkey]+ratecuttrigramdict[pkey])
                ratecutcount += ratecuttrigramdict[pkey]/(pval+nohiketrigramdicteqtocut[pkey]+ratecuttrigramdict[pkey])
    elif pkey in nohiketrigramdicteqtocut:
        if pkey not in ratecuttrigramdict:
            nohikecount += 1 # trigram only in hiketrigramdict
        elif pkey in ratecuttrigramdict:
            nohikecount += nohiketrigramdicteqtocut[pkey]/(nohiketrigramdicteqtocut[pkey]+ratecuttrigramdict[pkey])
            ratecutcount += ratecuttrigramdict[pkey]/(nohiketrigramdicteqtocut[pkey]+ratecuttrigramdict[pkey]) 
    elif pkey in ratecuttrigramdict:
        ratecutcount += 1 # trigram only in ratecuttrigramdict
    else:
        nomatch += 1

hikefrac = hikecount/(hikecount+nohikecount+ratecutcount)
nohikefrac = nohikecount/(hikecount+nohikecount+ratecutcount)
ratecufrac = nohikecount/(hikecount+nohikecount+ratecutcount)
hikefracfull = hikecount/(hikecount+nohikecount+ratecutcount+nomatch)
nohikefracfull = nohikecount/(hikecount+nohikecount+ratecutcount+nomatch)
ratecutfrac = ratecutcount/(hikecount+nohikecount+ratecutcount+nomatch)
nomatchfrac = nomatch/(hikecount+nohikecount+ratecutcount+nomatch)

print(hikecount, ":hikecount, ", nohikecount, ":nohikecount, ", ", ", ratecutcount, ":ratecutcount", ", ", 
      nomatch, ":nomatch", ", ")
print(hikefrac, ":hikefrac, ", nohikefrac, ":nohikefrac, ", ", ")
print(hikefracfull, ":hikefracfull, ", nohikefracfull, ":nohikefracfull, ",  ratecutfrac, ":ratecutfrac, ", 
      nomatchfrac, ":nomatchfrac")
if hikecount > nohikecount:
    print(": Rate Hike")
else:
    print(": No Rate Hike")

# count for frequency for trigram match considering unique or common occurence across 3 file sets
for pkey, pval in predict_trigram.items():
    if pkey in hiketrigramdicteqtocut:
        if pkey not in nohiketrigramdicteqtocut:
            if pkey not in ratecuttrigramdict:
                hikecount += pval # trigram only in hiketrigramdicteqtocut
            elif pkey in ratecuttrigramdict:
                hikecount += pval
                ratecutcount += ratecuttrigramdict[pkey]
        elif pkey in nohiketrigramdicteqtocut:
            if pkey not in ratecuttrigramdict:
                hikecount += pval
                nohikecount += nohiketrigramdicteqtocut[pkey]
            elif pkey in ratecuttrigramdict:
                hikecount += pval
                nohikecount += nohiketrigramdicteqtocut[pkey]
                ratecutcount += ratecuttrigramdict[pkey]
    elif pkey in nohiketrigramdicteqtocut:
        if pkey not in ratecuttrigramdict:
            nohikecount += nohiketrigramdicteqtocut[pkey] # trigram only in hiketrigramdict
        elif pkey in ratecuttrigramdict:
            nohikecount += nohiketrigramdicteqtocut[pkey]
            ratecutcount += ratecuttrigramdict[pkey]
    elif pkey in ratecuttrigramdict:
        ratecutcount += ratecuttrigramdict[pkey] # trigram only in ratecuttrigramdict
    else:
        nomatch += 1

hikefrac = hikecount/(hikecount+nohikecount+ratecutcount)
nohikefrac = nohikecount/(hikecount+nohikecount+ratecutcount)
ratecufrac = nohikecount/(hikecount+nohikecount+ratecutcount)
hikefracfull = hikecount/(hikecount+nohikecount+ratecutcount+nomatch)
nohikefracfull = nohikecount/(hikecount+nohikecount+ratecutcount+nomatch)
ratecutfrac = ratecutcount/(hikecount+nohikecount+ratecutcount+nomatch)
nomatchfrac = nomatch/(hikecount+nohikecount+ratecutcount+nomatch)

print(hikecount, ":hikecount, ", nohikecount, ":nohikecount, ", ", ", ratecutcount, ":ratecutcount", ", ", 
      nomatch, ":nomatch", ", ")
print(hikefrac, ":hikefrac, ", nohikefrac, ":nohikefrac, ", ", ")
print(hikefracfull, ":hikefracfull, ", nohikefracfull, ":nohikefracfull, ",  ratecutfrac, ":ratecutfrac, ", 
      nomatchfrac, ":nomatchfrac")
if hikecount > nohikecount:
    print(": Rate Hike")
else:
    print(": No Rate Hike")
    
