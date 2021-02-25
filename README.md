# WiDS2021

<a href="https://www.kaggle.com/c/widsdatathon2021"> Kaggle WiDS 2021 competition</a> 

Some code used for the submissions I've made to the WiDS 2021 competition on Kaggle.

### Python libraries used
pandas, numpy, catboost, lightgbm, hyperopt

### Motivation for the project
First Kaggle competition, with a friend (InesT).

### Files descriptions
- kaggle_setup.py : how to download data from kaggle via Kaggle API for python
- setup.py : needed packages and functions
- WiDS_v1_catboost_RS.ipynb : first submit, catBoost model with random search for parameter tunning
- WiDS_lightGBM_hyperopt_numerical_feat.ipynb : lightGBM model using only numerical features, parameters tunning with hyperopt 

### Conclusions of the project

Scores obtained on the unlabelled data:
- submission_catboost_RS_180221.csv : 0.84240
