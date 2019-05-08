#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 12:04:23 2019

@author: nitinsinghal
"""

# Using Classification algos to find category of rate hike/no hike / cut in FOMC statements 
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier, DecisionTreeClassifier

lemmatizer = WordNetLemmatizer()

stopwords = set(w.rstrip() for w in open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/NLPCourse/machine_learning_examples-master/nlp_class/stopwords.txt'))

# add more stopwords to remove from the text
stopwords = stopwords.union({'federal', 'fund', 'staff', 'reported', 'committee', 'discussed', 'desk', 'dealer', 
                            'meeting', 'agency', 'mortgage-backed', 'security', 'minute', 'board', 'forecaster', 
                            'governor', 'president', 'mexican', 'north', 'technology', 'example', 'addendum', 
                            'corporation', 'interbank', 'technological', 'requirement', 'seasonality'})

# Load FOMC statement documents
fomcdec99text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCDec99Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov99text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCNov99Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcoct99text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCOct99Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug99text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCAug99Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun99text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJun99Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay99text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMay99Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar99text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMar99Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcfeb99text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCFeb99Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec00text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCDec00Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov00text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCNov00Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcoct00text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCOct00Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug00text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCAug00Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun00text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJun00Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay00text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMay00Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar00text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMar00Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcfeb00text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCFeb00Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec01text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCDec01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov01text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCNov01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcoct01text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCOct01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep01text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCSep01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug01text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCAug01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun01text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJun01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay01text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMay01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcapr01text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCApr01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar01text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMar01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomc31jan01text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMC31Jan01Stmt.txt', encoding='utf-8', errors='ignore').read()
fomc03jan01text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMC03Jan01Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec02text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCDec02Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov02text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCNov02Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep02text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCSep02Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug02text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCAug02Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun02text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJun02Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay02text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMay02Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar02text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMar02Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan02text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJan02Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec03text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCDec03Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcoct03text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCOct03Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep03text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCSep03Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug03text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCAug03Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun03text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJun03Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay03text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMay03Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar03text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMar03Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan03text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJan03Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec04text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCDec04Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov04text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCNov04Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep04text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCSep04Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug04text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCAug04Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun04text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJun04Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay04text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMay04Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar04text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMar04Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan04text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJan04Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec05text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCDec05Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov05text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCNov05Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep05text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCSep05Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug05text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCAug05Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun05text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJun05Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay05text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMay05Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar05text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMar05Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcfeb05text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCFeb05Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec06text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCDec06Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcoct06text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCOct06Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep06text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCSep06Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug06text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCAug06Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun06text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJun06Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay06text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMay06Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar06text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMar06Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan06text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJan06Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec07text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCDec07Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcoct07text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCOct07Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep07text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCSep07Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug07text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCAug07Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun07text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJun07Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay07text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMay07Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar07text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMar07Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan07text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJan07Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec08text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCDec08Stmt.txt', encoding='utf-8', errors='ignore').read()
fomc8oct08text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMC8Oct08Stmt.txt', encoding='utf-8', errors='ignore').read()
fomc29oct08text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMC29Oct08Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep08text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCSep08Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug08text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCAug08Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun08text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJun08Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcapr08text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCApr08Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar08text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMar08Stmt.txt', encoding='utf-8', errors='ignore').read()
fomc22jan08text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMC22Jan08Stmt.txt', encoding='utf-8', errors='ignore').read()
fomc30jan08text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMC30Jan08Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec09text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCDec09Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov09text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCNov09Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep09text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCSep09Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug09text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCAug09Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun09text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJun09Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcapr09text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCApr09Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar09text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMar09Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan09text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJan09Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec14text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCDec14Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcoct14text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCOct14Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep14text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCSep14Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjul14text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJul14Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun14text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJun14Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar14text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMar14Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcapr14text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCApr14Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan14text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJan14Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec15text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCDec15Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcoct15text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCOct15Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep15text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCSep15Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjul15text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJul15Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun15text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJun15Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar15text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMar15Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcapr15text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCApr15Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan15text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJan15Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec16text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCDec16Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov16text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCNov16Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep16text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCSep16Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjul16text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJul16Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun16text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJun16Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar16text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMar16Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcapr16text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCApr16Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan16text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJan16Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec17text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCDec17Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov17text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCNov17Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep17text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCSep17Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjul17text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJul17Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun17text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJun17Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay17text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMay17Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar17text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMar17Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcfeb17text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCFeb17Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcdec18text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCDec18Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcnov18text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCNov18Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcsep18text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCSep18Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcaug18text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCAug18Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjun18text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJun18Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmay18text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMay18Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcmar18text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCMar18Stmt.txt', encoding='utf-8', errors='ignore').read()
fomcjan18text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJan18Stmt.txt', encoding='utf-8', errors='ignore').read()

fomcjan19text = open('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/MachineLearningCode/NLPTestData/FOMCJan19Stmt.txt', encoding='utf-8', errors='ignore').read()


ratehikefiles = [fomcmar18text, fomcjun18text, fomcsep18text, fomcdec17text, fomcjun17text, fomcmar17text, 
                 fomcdec15text, fomcdec18text, fomcdec05text, fomcnov05text, fomcsep05text, fomcaug05text, 
                 fomcjun05text, fomcmay05text, fomcmar05text, fomcfeb05text, fomcjan06text, fomcmar06text, 
                 fomcmay06text, fomcjun06text, fomcjun04text, fomcaug04text, fomcsep04text, fomcnov04text, 
                 fomcdec04text, fomcjun99text, fomcaug99text, fomcnov99text, fomcfeb00text, fomcmar00text, 
                 fomcmay00text, fomcdec16text] #
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


# create a function to tokenize text. Remove short words less than 2 alphabets
# then convert to tagged words for hike/no hike/cut
def text_tokenize(texttoken):
    txt = texttoken
    txt = txt.lower()
    tokens = nltk.tokenize.word_tokenize(txt)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    return tokens

# As there are more noratehike, ratehike files compared to ratecut files, we will take random sample 
# of noratehike, ratehike = ratecut files
# This ensures that we are comparing equal no of files to create equal word token sets for feautres
np.random.shuffle(noratehikefiles)
noratehikefileseqtocut = noratehikefiles[:len(ratecutfiles)]
np.random.shuffle(ratehikefiles)
ratehikefileseqtocut = ratehikefiles[:len(ratecutfiles)]

taggedwords = []
for hfile in ratehikefileseqtocut:
    taggedwords += [(list(text_tokenize(hfile)),categories[0])]
for nhfile in noratehikefileseqtocut:
    taggedwords += [(list(text_tokenize(nhfile)),categories[1])]
for cutfile in ratecutfiles:
    taggedwords += [(list(text_tokenize(cutfile)),categories[2])]
np.random.shuffle(taggedwords)

allfilewords = []
for hfile in ratehikefileseqtocut:
    allfilewords += text_tokenize(hfile)
for nhfile in noratehikefileseqtocut:
    allfilewords += text_tokenize(nhfile)
for cutfile in ratecutfiles:
    allfilewords += text_tokenize(cutfile)

# Using frequency distribution of the words, use top 1000 as that should be enough
all_words = nltk.FreqDist(t for t in allfilewords)
word_features = list(all_words)[:2000]

def document_features(document): 
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# Create the tagged word feature set and split into train/test
featuresets = [(document_features(d), c) for (d,c) in taggedwords]
train_set, test_set = featuresets[10:], featuresets[:10]

#predict_text = fomcnov02text #ratecut
#predict_text = fomcjun16text #nohike
#predict_text = fomcdec16text #hike
predict_text = fomcjan19text

# Apply Naive Bayes classifier
NBclassifier = NaiveBayesClassifier.train(train_set)
print('NaiveBayes Accuracy: ', nltk.classify.accuracy(NBclassifier, test_set))
NBclassifier.show_most_informative_features(20)

NBpredict_set = document_features(text_tokenize(predict_text))
print('NaiveBayes Prediction: ', NBclassifier.classify(NBpredict_set))

# Apply DecisionTree classifier
DTclassifier = DecisionTreeClassifier.train(train_set)
print('DecisionTree Accuracy: ', nltk.classify.accuracy(DTclassifier, test_set))

DTpredict_set = document_features(text_tokenize(predict_text))
print('DecisionTree Prediction: ', DTclassifier.classify(DTpredict_set))


########Trigram Classification - Not giving good results ##########
#def token_trigram(fomctext): 
#    tritokens = text_tokenize(fomctext)
#    trigrams = {}    
#    for i in range(len(tritokens) - 2):
#        k = (tritokens[i], tritokens[i+1], tritokens[i+2])
#        if k not in trigrams:
#            trigrams[k] = 1
#        else:
#            trigrams[k] += 1
#    return trigrams
#
#taggedtrigrams = []
#for hfile in ratehikefileseqtocut:
#    taggedtrigrams += [(list(token_trigram(hfile)),categories[0])]
#for nhfile in noratehikefileseqtocut:
#    taggedtrigrams += [(list(token_trigram(nhfile)),categories[1])]
#for cutfile in ratecutfiles:
#    taggedtrigrams += [(list(token_trigram(cutfile)),categories[2])]
#np.random.shuffle(taggedtrigrams)
#
#allfiletrigrams = []
#for hfile in ratehikefileseqtocut:
#    allfiletrigrams += token_trigram(hfile)
#for nhfile in noratehikefileseqtocut:
#    allfiletrigrams += token_trigram(nhfile)
#for cutfile in ratecutfiles:
#    allfiletrigrams += token_trigram(cutfile)
#
## Using frequency distribution of the trigrams, use top 1000 as that should be enough
#all_trigrams = nltk.FreqDist(t for t in allfiletrigrams)
#trigram_features = list(all_trigrams)[:2000]
#
#def document_features(document): 
#    document_trigram = set(document)
#    features = {}
#    for trigram in trigram_features:
#        features['contains({})'.format(trigram)] = (trigram in document_trigram)
#    return features
#
## Create the tagged trigrams feature set and split into train/test
#featuresets = [(document_features(t), c) for (t,c) in taggedtrigrams]
#train_set, test_set = featuresets[10:], featuresets[:10]
#
##predict_text = fomcnov02text #ratecut
##predict_text = fomcjun16text #nohike
#predict_text = fomcdec16text #hike
##predict_text = fomcjan19text
#
## Apply Naive Bayes classifier
#NBclassifier = NaiveBayesClassifier.train(train_set)
#print('NaiveBayes Accuracy: ', nltk.classify.accuracy(NBclassifier, test_set))
#NBclassifier.show_most_informative_features(20)
#
#NBpredict_set = document_features(text_tokenize(predict_text))
#print('NaiveBayes Prediction: ', NBclassifier.classify(NBpredict_set))
#
## Apply DecisionTree classifier
#DTclassifier = DecisionTreeClassifier.train(train_set)
#print('DecisionTree Accuracy: ', nltk.classify.accuracy(DTclassifier, test_set))
#
#DTpredict_set = document_features(text_tokenize(predict_text))
#print('DecisionTree Prediction: ', DTclassifier.classify(DTpredict_set))
