import numpy as np
import sklearn
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier



df = pd.read_csv('doc/nlp_otr/data/train.txt',header= None)
documents = df[0].values
#
# df_target = pd.read_csv('data/labels.txt',header=None)
# targets = df_target.values



def sklearn_vectorizer(documents):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(documents)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    return X_train_tf

def get_truelist(documents,df_target,X_test_f,clf):

    skvect = sklearn_vectorizer(documents)


    X_train,X_test,y_train,y_test = train_test_split(skvect,df_target[0].values,test_size=0.35,random_state=1234)

    clf.fit(X_train,y_train)
    predicted = clf.predict(X_test)

    true_idx = np.argwhere(predicted)
    ltrue = list(true_idx[:,0])

    truelist = pd.DataFrame(X_test_f).loc[ltrue]


    return truelist




def main(documents=documents):
    # home = '/perm/chatproj/fitnessdata/'

    #
    # df = pd.read_csv('data/train.txt',header= None)
    # documents = df[0].values

    df_target = pd.read_csv('doc/nlp_otr/data/labels.txt',header=None)
    targets = df_target.values


    clf= SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)

    X_train_f,X_test_f,y_train_f,y_test_f = train_test_split(documents,df_target[0].values,test_size=0.35,random_state=1234)





    return get_truelist(documents,df_target,X_test_f,clf)



"""

get top features from the marked-true documents
todo: eliminate redundant processes

"""


from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')





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






def get_top_words(truelist,documents=documents):
    stop = set(stopwords.words('english'))
    cv=CountVectorizer(max_df=0.85,stop_words=stop)
    word_count_vector=cv.fit_transform(documents)

    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    truelist.columns=['content']

    feature_names=cv.get_feature_names()

    keyword_list = []
    for doc in truelist['content']:
        tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

        #sort the tf-idf vectors by descending order of scores
        sorted_items=sort_coo(tf_idf_vector.tocoo())

        #extract only the top n; n here is 10
        keywords=extract_topn_from_vector(feature_names,sorted_items,3)

        keyword_list.append(keywords)


    return keyword_list
