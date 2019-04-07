import numpy as np
import sklearn
import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
random_seed = np.random.randint(1,500)
from sklearn import metrics



infile_corpus = open('doc/nlp_otr/corpus','rb')
corpus = pickle.load(infile_corpus)
infile_corpus.close()


infile_targets = open('doc/nlp_otr/targets','rb')
targets = pickle.load(infile_targets)
infile_targets.close()

#
#

"""

clf= SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=random_seed, max_iter=5, tol=None)
# with open('modelnew','rb') as file:
#     clf = pickle.load(file)


# def get_truelist_small(df,clf=clf,random_seed=random_seed):
#     # documents = df[0].values

#     skvect = sklearn_vectorizer(documents)

#     clf.predict(documents)

#     true_idx = np.argwhere(predicted)
#     ltrue = list(true_idx[:,0])

#     truelist = documents.loc[ltrue]
#     return truelist


def sklearn_vectorizer(documents):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(documents)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    return X_train_tf


def train_model(documents,df_target,clf=clf,random_seed=random_seed):

    skvect = sklearn_vectorizer(documents)
    X_train,X_test,y_train,y_test = train_test_split(skvect,df_target,test_size=0.35,random_state=random_seed)

    clf.fit(X_train,y_train)


    predicted = clf.predict(X_test)
    score = np.mean(predicted == y_test)
    print('prediction score:\t', score,'\n')
    print(metrics.classification_report(y_test, predicted, target_names=['fitness','other']))
    print('confusion:\n', np.array([['TP','FN'],['FP','TN']]),'\n', metrics.confusion_matrix(y_test, predicted).T)

def get_truelist(documents,df_target,X_test_f,clf):

    skvect = sklearn_vectorizer(documents)


    X_train,X_test,y_train,y_test = train_test_split(skvect,df_target,test_size=0.35,random_state=random_seed)

    clf.fit(X_train,y_train)
    predicted = clf.predict(X_test)

    true_idx = np.argwhere(predicted)
    ltrue = list(true_idx[:,0])

    truelist = pd.DataFrame(X_test_f).loc[ltrue]


    return truelist




def main(documents=corpus,targets=targets,clf=clf):

#     clf= SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)

    X_train_f,X_test_f,y_train_f,y_test_f = train_test_split(documents,targets,test_size=0.35,random_state=1234)

    train_model(documents,targets,clf=clf,random_seed=random_seed)




    return get_truelist(documents,targets,X_test_f=X_test_f,clf=clf)

# def main(documents=corpus):


#     return get_truelist_small(documents)



"""



import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

#
#
# try:
#     nltk.data.find('stopwords')
# except LookupError:
#     nltk.download('stopwords')

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]

    return results

def get_top_words(truelist,documents=corpus):
    stop = stopwords
    cv=CountVectorizer(max_df=0.85,stop_words=stop)
    word_count_vector=cv.fit_transform(documents)

    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)

#     truelist.columns=['content']

    feature_names=cv.get_feature_names()

    keyword_list = []
#     for doc in truelist['content']:
    for doc in truelist:
        tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

        #sort the tf-idf vectors by descending order of scores
        sorted_items=sort_coo(tf_idf_vector.tocoo())

        #extract only the top n; n here is 10
        keywords=extract_topn_from_vector(feature_names,sorted_items,3)

        keyword_list.append(keywords)


    return keyword_list
