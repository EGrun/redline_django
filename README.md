# redline_nlprework_dev


Reworking of my NLP module for the Yuuvis hackathon hosted by Optimal Systems GmbH.


This module now relies exclusively on SpaCy and GenSim for classification and tagging.


The included mess of files are the SpaCy classifier, which has been pre-trained on the included test data.
This means that this module _will not_ be applicable across datasets without retraining.




## FOR AN EXAMPLE OF HOW TO USE and MORE DOCUMENTATION ON THE PROCESS:

See comments in main_2.py


### To Do:

Reduce the redundant operations between classification_model_v1 (cm1) and keyword_generator_internal_v1 (kgi1)

