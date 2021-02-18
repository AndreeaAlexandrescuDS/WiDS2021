### packages
# dataframes manipulations
import pandas as pd
# numerics
import numpy as np
#data inspection
import dabl

import kaggle

#notebook displays
from IPython.display import display_html, Image

#vizuals
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import shap
from mpl_toolkits.mplot3d import Axes3D

#models
import catboost
from catboost import Pool, cv, CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit 
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, roc_curve, auc, precision_score           
from sklearn.metrics import precision_recall_curve, recall_score, f1_score, accuracy_score, roc_auc_score, log_loss
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import manifold

# params tunning
# from hyperopt import Trials, STATUS_OK, hp, tpe, fmin

#garbage collector
import gc
#warnings
import warnings
warnings.filterwarnings("ignore")


#pandas settings
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:20,.2f}'.format)
pd.set_option('display.max_colwidth', -1)
np.random.seed(32)
random_state = 32


### functions

def value_cnts(df, cat_vars):
    '''returns value counts of categorical variables cat_vars(one or many)'''
    df_cnt = df[cat_vars].value_counts().reset_index().sort_values(by = 0, ascending=False).rename(columns={0:'count'})
    df_cnt['% of total'] = round(df_cnt['count']/len(df)*100, 2)
    return df_cnt

def display_side_by_side(*args):
    '''diplay two or more dataframes side-by-side'''
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

## find NAs  
def inspect_missing_data(df):  
    '''find NAs for each column of a dataframe'''
    '''return head&tail of columns with NAs'''
    missing_data = df.isna().sum().reset_index().sort_values(by=0, ascending=False)
    no_missing = missing_data[missing_data[0] != 0].shape[0]
    total_cols = df.shape[1]
    total_rows = df.shape[0]
    
    missing_data.columns = ["name", "missing appearences"]
    missing_data["%missing from total"] = (missing_data[missing_data["missing appearences"]!=0]["missing appearences"]/total_rows)*100
    
    too_much_miss = missing_data[missing_data["%missing from total"] > 80].shape[0]
    to_drop = missing_data[missing_data["%missing from total"] > 80]["name"].to_list()
    
    print("There are {}/{} columns with missing data.".format(no_missing, total_cols))
    print("There are {}/{} columns with more than 80% missing data".format(too_much_miss, no_missing))
    print("Features with largest/smallest percent of missing values(top 10): ")
    
    tail = round(missing_data.tail(10).sort_values(by='%missing from total'), 2)
    head = round(missing_data.head(10).sort_values(by='%missing from total'), 2) 
       
    return display_side_by_side(head, tail)

## plot features importance for tree model
def plot_feat_imp(model):
    feature_importance_df = pd.DataFrame(model.get_feature_importance(prettified=True))
    plt.figure(figsize=(10, 30));
    sns.barplot(x="Importances", y="Feature Id", data=feature_importance_df);
    plt.title('Feature importance', fontsize=16, weight="bold"); 
    
def load_patch(image_id, coor, size=size):
    """ Load images from Jupyter Notebook """
    images = save_patch(image_id, coor, size)
    html = (f'<a href="{images[0]}" target="_blank">{images[0]}</a><br>'
            f'<a href="{images[1]}" target="_blank">{images[1]}</a><br>'
            f'<a href="{images[2]}" target="_blank">{images[2]}</a>')
    return HTML(html)    
