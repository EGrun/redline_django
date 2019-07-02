
import json
import numpy as np
import gensim
import spacy
import io, os, glob, sys, re
from spacy.lang.en import English
from collections import defaultdict, Counter
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from heapq import nlargest
from time import time


__docs__ = 'keyword generator. \n optional save_json flag on kg.main(), which, \n when set to True, will output the generated json as a file. \n By default, simply returns json as an object'


def keep_token(t):
    return (t.is_alpha and (t.pos_ in ['PROPN','NOUN']) and not (t.is_space or t.is_punct or t.is_stop or t.like_num))

def lemmatize_doc(doc):
    return [t.lemma_ for t in doc if keep_token(t)]

def prepare_docs(input_list):

    nlp = spacy.load('en_core_web_sm')

    token_vocab=[]
    sentences =[]
    for text in input_list:
        doc = nlp(text.text)
        token_vocab.append(lemmatize_doc(doc))
        for sent in doc.sents:
            sentences.append(sent.text)

    num_docs = len(token_vocab) #num_docs will be reused multiple times

    """creating gensim dictionary and corpus for the nouns"""
    dictionary = corpora.Dictionary(token_vocab)
    # dictionary.save('/tmp/docdict.dict') #uncomment to save dict for document reanalysis
    corpus = [dictionary.doc2bow(text) for text in token_vocab]
    # corpora.MmCorpus.serialize('/tmp/corpus.mm',corpus) #uncomment to save corpus as Mmatrix
    """activating tfidf analysis on corpus"""
    tfidf = gensim.models.TfidfModel(corpus,normalize=True)
    corpus_tfidf = tfidf[corpus]

    return corpus_tfidf,corpus,num_docs,dictionary,sentences

def turn_corpus_into_useful_array(num_docs,corpus_to_use,dictionary):
    """
    with this method, we attach the original vocab word to either the tfidf score-entry or the frequency score-entry.
    inputs:
    For this demonstration:
    for tfidf, var corpus_to_use = corpus_tfidf
    for frequency, var corpus_to_use = corpus

    num_docs = the number of documents in question, which for this demo has been predefined as num_docs

    output:
    np array with the requisite words and scores
    """
    result_with_vocab_base=[ [] for i in range(num_docs)]
    for ind,doc in enumerate(corpus_to_use):
        result_with_vocab_base[ind] = sorted(doc,key=lambda i: i[1], reverse=True)
        #probably can eliminate this redundant step, but need to convert from native tuple to list for appending vocab text

    result_with_vocab = [ [] for i in range(num_docs)]

    for ind,doc in enumerate(result_with_vocab_base):
        for entry in doc:
            result_with_vocab[ind].append(list([*entry,dictionary[entry[0]]]))

    result = np.array(result_with_vocab)
    return result


def fetch_top_words(documentnumber, model_df,n=3):
    """
    this convenience function will:

    output the n top-scoring words for the scoring model in question.

    the function simply takes the values of the first three words for each document, so any sorted model output can be used.

    """

    topwords = []
    for i in model_df[documentnumber][0:n]:
        topwords.append(i[2])
    return topwords


def get_sentence_scores(tags,sentences):
    # just_tags =[]
    # for t,_ in tags:
    #     just_tags.append(t)
    just_tags =[t for t,_ in tags]
    important_sentences = {word: {} for word in just_tags}

    for word in just_tags:
        for sent in sentences:
            #sent score will estimate the relevancy of a sentence to a keyword
            if word in sent:
                important_sentences[word].update({sent:0})

    for word,sentences_and_scores in important_sentences.items():
            for entry,score in sentences_and_scores.items():
                sent_score = 0
                for w in entry.split(' '):
                    if w in just_tags:
                        sent_score+=1

                sentences_and_scores[entry]=sent_score
    return important_sentences





def _assess_top_words(num_docs,n,model_list,sentences,outdict,docnum):

    model_outputs = {}
    for model_output in model_list:
        indv_output = fetch_top_words(docnum,model_output,n)
        for ind, el in enumerate(indv_output):
            if el not in model_outputs:
                model_outputs[el] = n-ind
            else:
                model_outputs[el]+=(n-ind)

    tags = sorted(model_outputs.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)

    # important_sentences = get_sentence_scores(tags,sentences)
    outdict[docnum] = []


    top_tags = [t for t,_ in tags][:5]
    for word in top_tags:
        outdict[docnum].append(word)

def create_outdict(num_docs,n,model_list,sentences,outdict):

    for docnum in range(num_docs):
        _assess_top_words(num_docs,n,model_list,sentences,outdict,docnum)



def _save_json(outdict):
    filename = 'json_output_{0:.0f}.json'.format(time())
    with open(filename,'w') as f:
        json.dump(outdict,f)
        f.close()
    print('file saved at ' + filename)

def main(input_list,save_json=False):
    corpus_tfidf,corpus,num_docs,dictionary,sentences = prepare_docs(input_list)
    df_freq = turn_corpus_into_useful_array(num_docs,corpus,dictionary)
    df_tfidf=turn_corpus_into_useful_array(num_docs,corpus_tfidf,dictionary)
    outdict = {}
    create_outdict(num_docs,7,[df_freq,df_tfidf],sentences,outdict=outdict)
    if save_json==True:
        _save_json(outdict)

    return json.dumps(outdict, skipkeys=True)
