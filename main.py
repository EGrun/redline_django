import classification_model_v1 as cm1
import keyword_generator_internal_v1 as kgi1



"""
runs the classification model, outputs that to the keyword generator, and outputs the keywords.

i/o sequence:
the data is stored in the directory specified by 'datadir':

for this demo, the data in question is pre-classified, and therefor there are 'train.txt' and  'labels.txt' files required by classification_model_v1.py. In the next iteration, this aspect will be removed to more closely resemble a production scenario.

The documents pass through the classification model, and the test-set documents which have been labelled as being of the positive class are then outputted as 'to_be_tagged'.

Because the original to_be_tagged file has about 330 entries, for demo purposes it has been reduced to the first 20. This is arbitrary.

The (shortened) to_be_tagged object is then passed to the keyword generator.
This generator parses for nouns, and then analyzes word frequency and tf-idf scoring and naively combines the two. Heuristically, the keywords outputted by this combination have been more representative of the data than relying on tf-idf alone.


The keywords are outputted to a json object, with format 'index number': {keyword list}.

The optional save_json parameter will, predictably, save the generated json as a json file if save_json=True.


"""
datadir = 'fitnessdata/'
modeldir='.'

def main(datadir,modeldir,save_json=False):
    testnlp = cm1.main(datadir,modeldir)

    to_be_tagged = cm1.return_to_be_tagged(testnlp)
    tbt_short = to_be_tagged[0:20]


    # kgi1.__docs__
    return kgi1.main(tbt_short,save_json)
