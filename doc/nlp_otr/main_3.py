import doc.nlp_otr.keys_and_file as rnlp
import pandas as pd
import numpy as np
import json
import pickle

def main():
    infile_corpus = open('doc/nlp_otr/corpus','rb')
    corpus = pickle.load(infile_corpus)
    infile_corpus.close()


    infile_targets = open('doc/nlp_otr/targets','rb')
    targets = pickle.load(infile_targets)
    infile_targets.close()



    k = rnlp.get_top_words(corpus)


    def pull_random(k=k,corpus=corpus):

        choice_id = np.random.randint(0,len(k))

        ids = k[choice_id]
        doc = corpus[choice_id]
        print(json.dumps((choice_id,doc,ids)))
        return json.dumps((choice_id,doc,ids))



    return pull_random()

# with open('data_oneline1.json', 'w') as outfile:
#         json.dump(pull_random(), outfile)
