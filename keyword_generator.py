
import pandas as pd
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

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def keep_token(t):
    return (t.is_alpha and (t.pos_ in ['PROPN','NOUN']) and not (t.is_space or t.is_punct or t.is_stop or t.like_num))

def lemmatize_doc(doc):
    return [t.lemma_ for t in doc if keep_token(t)]

def prepare_docs():

    nlp = spacy.load('en_core_web_sm')

    token_vocab=[]
    sentences =[]
    for file in glob.glob(os.path.join("data", '*.txt')):
        with open(file,'r') as file_:
            text=file_.readlines()
            doc = nlp(text)
            token_vocab.append(lemmatize_doc(doc))
            sentences.append(sent.text for sent in doc.sents)
            for sent in doc.sents:
                sentences.append(sent.text)


    #num_docs will be reused multiple times
    num_docs = len(token_vocab)

    """creating gensim dictionary and corpus for the nouns"""
    dictionary = corpora.Dictionary(token_vocab)
    # dictionary.save('/tmp/docdict.dict')
    corpus = [dictionary.doc2bow(text) for text in token_vocab]
    # corpora.MmCorpus.serialize('/tmp/corpus.mm',corpus)
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


def get_docnum(num_docs):
    print('\n')
    selection = input("which document (number 1-6) would you like to generate hashtags for? ")
    while selection not in [str(i) for i in (range(1,num_docs+1))]:
        print('input not recognized or out of range, please select again \n')
        selection = input("which document (number 1-6) would you like to generate hashtags for? ")

    return int(selection)


def get_sentence_scores(tags,sentences):
    # just_tags =[]
    # for t,_ in tags:
    #     just_tags.append(t)
    just_tags =[t for t,_ in tags]
    important_sentences = {word: {} for word in just_tags}

    for word in just_tags:
        for sent in sentences[1:]:
            #sent score will estimate the relevancy of a sentence to a keyword
            if word in sent:
                important_sentences[word].update({sent:0})

    for word,sentences_and_scores in important_sentences.items():
            for entry,score in sentences_and_scores.items():
                sent_score = 0
                for w in entry.split(' '):
                    # print(w)
                    if w in just_tags:
                        # print(w)
                        sent_score+=1
                        # print(sent_score)

                sentences_and_scores[entry]=sent_score
    return important_sentences


def save_data(saved_data):
    # global saved_data
    print('saving hashtags')
    try:
        outF = open('saved_hashtags.txt','a')
    except FileNotFoundError:
        outF = open('saved_hashtags.txt','w')
    for index, entry in enumerate(saved_data):
        if entry:
            outF.write('Document number: ')
            outF.write(str(index+1))
            outF.write('\n')

            for line in entry:
                outF.write(str(line))
                outF.write('\n')
    outF.write('\n\n\n')
    outF.close()




def _assess_top_words(num_docs,n,model_list,sentences,outdict,docnum):

    """
    internal version of above assess_top_words method which is used by below save_csv method.
    """

    model_outputs = {}
    for model_output in model_list:
        indv_output = fetch_top_words(docnum,model_output,n)
        for ind, el in enumerate(indv_output):
            if el not in model_outputs:
                model_outputs[el] = n-ind
            else:
                model_outputs[el]+=(n-ind)

    tags = sorted(model_outputs.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)

    important_sentences = get_sentence_scores(tags,sentences)


    top_tags = [t for t,_ in tags][:5]
    for word, sentence_list in important_sentences.items():
        if word in top_tags:
            relevant_sentences = nlargest(10,sentence_list, key=sentence_list.get)
            outdict[word] = []
            for sent in relevant_sentences:
                outdict[word].extend([sent,int(sentence_list.get(sent)) * 100,docnum])


def save_df_as_json(num_docs,n,model_list,sentences,outdict):
    """I was trying to avoid loading pandas for this script, but for outputting csvs, you can't beat it"""

    for docnum in range(num_docs):
        _assess_top_words(num_docs,n,model_list,sentences,outdict,docnum)
    df = pd.DataFrame.from_dict(outdict,orient='index')
    df.to_json('results_json.json')

    return df.to_json()

def main():
    corpus_tfidf,corpus,num_docs,dictionary,sentences = prepare_docs()
    df_freq = turn_corpus_into_useful_array(num_docs,corpus,dictionary)
    df_tfidf=turn_corpus_into_useful_array(num_docs,corpus_tfidf,dictionary)
    saved_data = [[] for i in range(num_docs)]
    assess_top_words(num_docs,7,[df_freq,df_tfidf],sentences,saved_data)
    save_data(saved_data)
    outdict = {}
    save_df_as_csv(num_docs,7,[df_freq,df_tfidf],sentences,outdict=outdict)
    sys.exit(0)


if __name__ == '__main__':
    print('loading document analysis...\n')
    main()
