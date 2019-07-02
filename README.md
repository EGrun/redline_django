# redline_nlprework_dev


Reworking of my NLP module for the Yuuvis hackathon hosted by Optimal Systems GmbH.


This module now relies exclusively on SpaCy and GenSim for classification and tagging.


The included mess of files are the SpaCy classifier, which has been pre-trained on the included test data.


FOR AN EXAMPLE OF HOW TO USE:

see test_container.py for a template on how to plug this in to your script.


FOR MORE DOCUMENTATION ON THE PROCESS:

See comments in main.py



### To Do:

Reduce the redundant operations between classification_model_v1 (cm1) and keyword_generator_internal_v1 (kgi1)

from cm1, refactor so as not to check for existing classification labels and assess model accuracy, to allow for entry of untagged data.

refactor cm1 into a class object.

refactor kg1 into a class object.

