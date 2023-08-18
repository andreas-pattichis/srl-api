#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# To install spacy, the NLP pipeline used in the script,
# please visit https://spacy.io/usage, select the hardware configuration of your server machine, select Dutch language,
# In the same page, you will see a few installation options, you can choose and follow the one that you prefer
library(reticulate)
path_to_python <- "~/anaconda3/bin/python"
use_python(path_to_python)
import spacy
import numpy as np
import pandas as pd
import csv
import re
import io
import glob
import os
import errno
from operator import add
from spacy.lang.nl.examples import sentences 


# In[ ]:


# Please make sure the spacy pipeline for Dutch
# is installed (for details: https://spacy.io/models/nl) - this should have been already done in the previous step,
# but it is still good to check
# Note: we use the model trained on a large corpus of texts, you may want to opt for a medium or small corpus instead,
# in case efficiency is an issue
nlp = spacy.load("nl_core_news_lg")
# nlp = spacy.load("en_core_web_lg")


# In[ ]:


# Create a path to essays folder
# Note: you may want to update this path to point to the folder on a server where
# the essay draft will be saved
path = '/Users/u583149/Downloads/Essays-3/*.txt'

# In[ ]:


# Import student essays (this should work with an individual essay, as well)
files = glob.glob(path) 
for name in files: 
    try: 
        with open(name) as f: 
            pass 
    except IOError as exc: 
        if exc.errno != errno.EISDIR: 
            raise


# Save from the file names, the usernames
basename_without_ext = [ ]
for i in range(len(files)):
  name = os.path.splitext(os.path.basename(files[i]))[0]
  basename_without_ext.append(name)


# Import source texts as objects
# Note: currently, it shows only a local path; you may want to update this path to point to the folder on server
# where your source texts (i.e., readings) will be hosted
# Each topic contains texts clumped together into one file, so we have 1 .txt file per topic
# (e.g., artificial intelligence, differentiation, scaffolding)
# ai = open('/Users/u583149/Downloads/Essays/source/AI.rtf', 'r')
# dif = open('/Users/u583149/Downloads/Essays/source/Differentiation.rtf', 'r')
# sc = open('/Users/u583149/Downloads/Essays/source/Scaffolding.rtf', 'r')
ai = open('/Users/u583149/Downloads/Essays_NL/AI_NL.rtf', 'r')
dif = open('/Users/u583149/Downloads/Essays_NL/Differentiatie.rtf', 'r')
sc = open('/Users/u583149/Downloads/Essays_NL/Scaffolding_NL.rtf', 'r')

# Build nlp objects for texts
doc_1 = nlp(ai.read())
doc_2 = nlp(dif.read())
doc_3 = nlp(sc.read())

# Fetch and process each essay
max_numberwords = 400
es_source_overlap = []
cohesion = []
mean_cohesion = []
word_count = []
essay_number = []
for i in range(len(files)):
    essay = open(files[i], 'r')
    essay_number.append(re.findall(r'\d+', re.search(r"(?<=pnig).*?(?=_essay.txt)", files[i]).group(0)))
    word_count.append(len(open(files[i], 'r+').read().split()))
    word_countrel= word_count[i]/ max_numberwords
    # Build nlp object for the essay
    doc_essay = nlp(essay.read())

    # Tokenize essay into sentences
    l=[]
    for sent in doc_essay.sents:
        l.append(sent)
        
# Compute semantic overlap with sources

    # Loop over each sentence and compute its semantic overlap with each source text
    source_overlap_1 = []
    source_overlap_2 = []
    source_overlap_3 = []
    for sent in range(len(l)):
        source_overlap_1.append(doc_1.similarity(l[sent]))
        source_overlap_2.append(doc_2.similarity(l[sent]))
        source_overlap_3.append(doc_3.similarity(l[sent]))
        
    # Semantic overlap with sources for the essay
    es_source_overlap.append((np.mean(source_overlap_1) + np.mean(source_overlap_2) + np.mean(source_overlap_3)))
    print(es_source_overlap)
    
# Compute sentence mean cohesion (combinations without repetition)
    for k in range(0,len(l)-1):
        for j in range(k+1,len(l)):
            cohesion.append(l[j].similarity(l[k]))
    # Mean cohesion for the essay
    mean_cohesion.append(np.mean(cohesion))


# In[ ]:


# The output values are:
# es_source_overlap - semantic overlap between the essay and the source texts
# mean_cohesion - mean sentence-to-sentence cohesion
# word_count - number of words in the essay
# The essay score may be computed as: es_source_overlap + mean_cohesion
#### One possible option for the script deployment:
# 1. embedd the script into the language used for dashboard
# 2. return the output values to dashboard pipeline
# 3. display the values in the dashboard
essay_score = np.add(es_source_overlap, mean_cohesion, word_countrel)



import csv  

header = ['First.name', 'Essay.score']
data = list(zip(basename_without_ext, essay_score))

with open('essay_score_NL.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    writer.writerows(data)

