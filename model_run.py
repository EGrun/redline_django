from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


output_dir = '.'

datadir = '/Users/sylvan/Documents/python_projects/chatproj/chatproj/fitnessdata'


df_text = pd.read_csv(datadir+'/train.txt',header=None)
df_cats = pd.read_csv(datadir+'/labels.txt',header=None)
df_cats.columns = ['is_fitness_app']
df_text.columns=['description']
cats = np.array(df_cats.is_fitness_app.apply(lambda y:{"POSITIVE": bool(y), "NEGATIVE": not bool(y)}))


xtrain,xtest,ytrain,ytest = train_test_split(df_text.description,df_cats.is_fitness_app)


print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)

testnlp = []
for i in xtest:
    testnlp.append(nlp2(i))
# doc2 = nlp2(xtest)


ri = ytest.reset_index()



accuracy = 0
# ri.is_fitness_app[:30]
for ind, el in enumerate(testnlp):
    assessment = max(el.cats.keys(), key=(lambda k: el.cats[k]))
    # el.cats,
    actual = ri.is_fitness_app[ind]

    if (assessment=='POSITIVE' and actual ==1):
        accuracy +=1
        
    elif (assessment=='NEGATIVE' and actual ==0):
        accuracy +=1



print(accuracy / len(testnlp))
