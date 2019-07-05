import classification_model_v2 as cm2



"""

Example of how to use the module.


runs the classification model, outputs that to the keyword generator, and outputs the keywords.

i/o sequence:

Inputs: data-storage directory ('datadir'), model-storage directory (should be root) ('modeldir')


the data is stored in the directory specified by 'datadir':

The documents pass through the classification model, and the test-set documents which have been labelled as being of the positive class are then outputted as 'to_be_tagged'.

Because the original to_be_tagged file in this demo has about 330 entries, for demo purposes it has been reduced to the first 20. This is arbitrary.

The (shortened) to_be_tagged object is then passed to the keyword generator.
This generator parses for nouns, and then analyzes word frequency and tf-idf scoring and naively combines the two. Heuristically, the keywords outputted by this combination have been more representative of the data than relying on tf-idf alone.


OUTPUT:

The keywords are outputted to a json object, with format 'index number': {keyword list}.

The optional save_json parameter will, predictably, save the generated json as a json file if save_json=True.

"""


"""
copy the below code, or a version thereof, into yours to get the desired output.

"""




datadir = 'fitnessdata'
modeldir='.'


classifier = cm2.Classifier(modeldir)

classifier.encode_documents(datadir)

tbt=classifier.return_to_be_tagged()

tbt_short = tbt[0:2]

kg = cm2.KeywordGen()
cm2.kg_main(kg,tbt_short)
