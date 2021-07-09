
import streamlit as st
import pandas as pd
import numpy as np
import pickle

#!/usr/bin/env python
# coding: utf-8

# # CU-Discomfort Project
# 
# ## Faculty of Allied Health Sciences, Chulalongkorn University
# 
#  All Rights Reserved.
#  
#  Author: Kao Panboonyuen, Ph.D.
#  
#  Year: 2021

#Importing the necessary libraries
import numpy as np
import pandas as pd
import math
import time
import pickle

from datetime import date, timedelta
#from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (12,8)




# Dimensionality reduction
import sklearn
from sklearn.decomposition import KernelPCA

# Data transformation classes
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.preprocessing import LabelEncoder as le
from sklearn.preprocessing import StandardScaler as ss
 
# Data splitting
from sklearn.model_selection import TimeSeriesSplit

# Model Pipelining
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ML Model
from scipy.stats import uniform, randint

from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

# from ngboost import NGBRegressor
# from ngboost.learners import default_tree_learner
# from ngboost.distns import Normal
# from ngboost.scores import MLE

# ML Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score, explained_variance_score

from sklearn.preprocessing import OneHotEncoder
import pickle

# from sklearn.ensemble import GradientBoostingRegressor


# # Useful Function



def encode_onehot(df, cols):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.
    
    Modified from: https://gist.github.com/kljensen/5452382
    
    Details:
    
    http://en.wikipedia.org/wiki/One-hot
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    vec = DictVectorizer()
    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df


# In[28]:


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100

def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


# In[29]:


def mape(y_test, y_pred):
    return np.mean(np.abs((y_test - y_pred) / y_test))

def evaluate(y_test, y_hat_dt):
    mae = mean_absolute_error(y_test, y_hat_dt)
    mse = mean_squared_error(y_test, y_hat_dt)
    rmse = mse*(1/2.0)

    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse) 
    
    print(mae)
    print(mse)
    print(rmse) 


# In[30]:


def transform_onehot(df, c=['Gender','DomSide','Period_Used']):
    with open('df_onehot.pkl', 'rb') as f:
        onehot_model = pickle.load(f)
    
    df_with_onehot = onehot_model.transform( df.loc[:, ['Gender','DomSide','Period_Used']] ).toarray()

    category_col = []
    for i in onehot_model.categories_:
        category_col = np.concatenate([category_col,i])
    print(category_col)

    #train[category_col] = train_with_onehot
    df[category_col] = df_with_onehot

    #train = train.select_dtypes('number')
    df = df.select_dtypes('number')
    
    return df


# ## Reverse Variables



# Reverse Variables 
onehote_col = ['Period_01',
       'Period_02', 'Period_03', 'Period_04', 'Period_05', 'Period_06',
       'Period_07', 'Period_08', 'Period_09', 'Period_10', 'Period_11',
       'Period_12', 'Period_13', 'Period_14', 'Period_15', 'Period_16',
       'Period_17', 'Period_18', 'Period_19', 'Period_20', 'Period_21',
       'Period_22', 'Period_23', 'Period_24']

def reverse_onehot(row):
    for c in onehote_col:
        if row[c]==1:
            return c
        
#test['Period'] = test.apply(reverse_onehot, axis=1)


# # to Streamlit

# In[32]:

st.header(
    'CU-Discomfort Forecasting Project'
)

# st.subheader(
#     'Guideline'
# )

# st.write(
#     """
#     In this project, we explore the possibility to apply machine learning to make diagnostic predictions from discomfort historical data.
#     """
# )

st.subheader('Faculty of Allied Health Sciences, Chulalongkorn University')
st.subheader(
    'To use model prediction, please following below steps:'
)

st.write(
    """
    1. From the left side of this page, there is an area to input several person conditions. \n
    2. To input person conditions that needed to be predicted. \n
    3. See the results below.
    """
)


# Create sidebar
st.sidebar.header(
    'Input person condition'
)

Name = st.sidebar.text_input(
            label = 'Please type your name',
            value= 'e.g., Messi'
        )

Gender = st.sidebar.selectbox(
    'Gender', ['Male','Female']
)

DomSide = st.sidebar.selectbox(
    'DomSide', ['Rt','Lt']
)

Age = st.sidebar.slider(
    'Please slide your age',
    1, 200, step = 1
)

weight = st.sidebar.slider(
    'Please slide your weight',
    1, 200, step = 1
)

height = st.sidebar.slider(
    'Please slide your height',
    1, 200, step = 1
)

Age = int(Age)
weight = int(weight)
height = int(height)

st.subheader(
    'Check your current information:'
)

st.write("Name: ", Name, type(Name))
st.write("Gender: ", Gender, type(Gender))
st.write("DomSide: ", DomSide, type(DomSide))

st.write("Age: ", Age, type(Age))
st.write("Weight: ", weight, type(weight))
st.write("Height: ", height, type(height))


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df)

# initialise data of lists.
Name = Name
Age = Age
Gender = Gender
weight = weight
height = height

BMI = weight / (height/100)**2

DomSide = DomSide

data = {'Name':[Name]*24,
       'Age':[Age]*24,
       'Gender':[Gender]*24,
       'BMI':[BMI]*24,
       'DomSide':[DomSide]*24,
       'Period_Used': ['Period_01', 'Period_02', 'Period_03', 
                  'Period_04', 'Period_05', 'Period_06', 'Period_07', 
                  'Period_08', 'Period_09', 'Period_10', 'Period_11', 
                  'Period_12', 'Period_13', 'Period_14', 'Period_15', 
                  'Period_16', 'Period_17', 'Period_18', 'Period_19', 
                  'Period_20', 'Period_21', 'Period_22', 'Period_23', 'Period_24']}



# Create DataFrame
df = pd.DataFrame(data)
df.set_index("Name", inplace = True)

st.write(" ----- INPUT -----")


st.write(df)


st.write(" ----- FORECASTING RESULT -----")

df = transform_onehot(df)

#model_gbt = GradientBoostingRegressor(random_state=0)
model_gbt = pickle.load(open('cudiscomfort_gbt_model_v1.pkl', 'rb'))

y_hat_gbt = model_gbt.predict(df.loc[:, df.columns != 'Label'])

forecasts = pd.DataFrame(y_hat_gbt, 
                           columns=['Forecast'])

show_attr = ['Age', 'BMI', 'Period',  'Forecast']

df['Period'] = df.apply(reverse_onehot, axis=1)
df['Forecast'] = forecasts.values

st.write(df[show_attr])



st.write(" ----- FORECASTING RESULT (GRAPH) -----")

# gca stands for 'get current axis'
ax = plt.gca()

#plt.xticks(rotation = 'vertical')
#plt.xticks(rotation=90)

df.plot(kind='line',x='Period',y='Forecast',ax=ax)
plt.xticks(rotation=90)

plt.show()

df2 = df.set_index("Period")
chart_data = df2[['Forecast']]
st.line_chart(chart_data)

st.write("Note: mae of this model is ~ 0.6659 on validation set")   
st.write("Author: Kao Panboonyuen, Contact: panboonyuen.kao@gmail.com")   
