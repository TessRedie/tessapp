from cmath import sqrt
from dataclasses import dataclass
from itertools import count
from math import log2
from multiprocessing import Pipe
from telnetlib import SB
from typing import Container, Text
from xml.etree.ElementInclude import include
import streamlit as st
import pandas as pd
from PIL import Image

from sklearn import datasets

import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt
import plotly.express as px 
import numpy as np
from sklearn.impute import SimpleImputer
from pathlib import Path


#libraries for text processing
import nltk
from nltk import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
import string 
import spacy
from nltk.corpus import stopwords
import collections
import requests
import re
from collections import Counter

# for encoding and transforming
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MaxAbsScaler, MinMaxScaler
from sklearn.dummy import DummyClassifier

#pipeline
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer, make_column_transformer

#Grid Search
from sklearn.model_selection import train_test_split,GridSearchCV 


#model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 


#Model metrics

from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score, ConfusionMatrixDisplay, f1_score, fbeta_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.preprocessing import FunctionTransformer

#import pymongo
from pymongo import MongoClient


#Connect to the server MongoDB
client = pymongo.MongoClient('mongodb://localhost:27017/')

db = client["streamlipro"]

#from sklearn.datasets import WineQT
header = st.container()
dataset = st.container()
eda = st.container()
visualization = st.container()
Preprocessing=st.container()
transformation=st.container()
Train_test_split=st.container()
Parameters = st.container()
model = st.container()
grid = st.container()
pipeline = st.container()
Metric_plot = st.container()
model_training = st.container()
pca_data=st.container()
#-------------------------------------------------------------
st.sidebar.subheader("Developer Profile")
st.sidebar.subheader("Tesfabirhan W. REDIE")
#st.sidebar.image(r"/home/tess/Documents/python_projects/streamlit_app/images/oie_png(1).png", width=None)
st.sidebar.write("Joining to the IT profession lately, I'm currenly studying to be an expert in Artificial Intelligence Application development, thanks to Greta Val-de-Loire and my school, Ecole Microsoft IA by Simplon. Always passionate by R&D, Studies, Data and Application Development, I believe that I have great capacity to integrate, intervene, strong in proposing, efficient in group work as well as love being in autonomy.", align_text='center')
#------------------------------------------
about_project = st.sidebar.selectbox("About the Project", ("Summary of the Project", "Motivation"))
st.sidebar.header("Dataset")
dataset_name = st.sidebar.selectbox("Select Dataset", ("Dolphins", "Wine Quality", "Iris", "Breast cancer", "Spam classifier"))
#Data_Visualization = st.sidebar.selectbox("Select plot", ("Pair Plot", "Violin Plot", "Correlation matrix", "3D Scatter Plot"))
classifier_name = st.sidebar.selectbox("Select model", ("KNN", "SVM", "Decision Tree Classifier", "Gradient Boosting Classifier", "Random Forest Classifier", "Random Forest Regressor"))

#dataset_model = st.sidebar.selectbox("Model dataset", ("Target, y","Features, X" ))
#-----------------------------------------
with header:
    st.title("Machine Learning and AI Application Models")
    if about_project =="Summary of the Project":
        st.header("About the project")
        st.write("In this ML project, I have tried to apply the skills I have learnt for the last 4 months. I have used public datasets. Originally from [Kaggle](https://www.kaggle.com/)Five datasets(Dolphins, Wine Quality, Iris, Breast cancer as well as Spam classification) are used for training three models(Random Forest Classifier, SVM, and KNN). For every dataset, Explanatory Data Analysis(EDA), Data Visualization, Features engineering, model training and evaluations, as well as PCA are presented. It\'s my first experience.  Its purpose is mainly to share knowlegde and learn from the feedbacks. Therefore feel free to add your own inputs.")
    
    elif about_project == "Motivation":
        st.subheader("Motivation")
        st.write("You may wonder what motivates me to do this project, am I right? If so, this is a brief statement")
        st.write("The popularity of Artificial Intelligence is soaring recently because of many factors. People are becoming more thirsty to train models so that a system or a program is able to think and learn from experiences and apply it for human benefits. AI applications are almost in every sector of our economy, social services and businesses. And, I'm not different. Joining the world of AI is giving me a unique experience. I have been dealing with data since early days of professional studies. I was trained to be a Surveying Engineer and then as an Agricultural Engineer. Data has been crucial throughout my studies. Nevertheless, it's only recently that I'm feeling more lean to data than ever before. This is because I found AI to be the best science to exploit data more efficiently for the intended purpose.")
        st.write("Saying that, my motivation to join the AI domain though in a haphazard way, I believe that it is an ambitious dream of more than two decades search for knowledge and skills pertaining to my passion and curiousty of data and data use. It's based on this curiousity driven endeavour that I feel encouraged to develop this ML website for boosting my professional sprint and I hope you will find it useful for you.")

with dataset:    
    if dataset_name == "Dolphins":
        st.subheader("1. Dolphins Dataset")
        #st.image(r"/home/tess/Documents/python_projects/streamlit_app/images/doplphin.png", width=None)
        st.markdown("[Source: Key West Aquarium](https://www.keywestaquarium.com/dolphins-in-key-west)")
        #data, dolphins
        #path
        
        mycollection = db['dolphins']
        st.write(mycollection)
        all_records = mycollection.find()
        list_cursor = list(all_records)
        
        data = pd.DataFrame(list_cursor, columns=['variety','area','dimension_1_mm', 'dimension_2_mm', 'dimension_3_mm', 'mass_g', 'sex'])
         #data type as float
        data['dimension_1_mm']  = pd.to_numeric(data.dimension_1_mm, errors='coerce')
        data['dimension_2_mm']  = pd.to_numeric(data.dimension_2_mm, errors='coerce')
        data['dimension_3_mm']  = pd.to_numeric(data.dimension_3_mm, errors='coerce') 
        data['mass_g']  = pd.to_numeric(data.mass_g, errors='coerce') 
        data = data.astype({"variety": str}, errors='raise') 
        
        data['variety'].astype(str)
        st.dataframe(data.style.highlight_max(axis=0))
    
#--------------------------------------------
    elif dataset_name == "Wine Quality":
        st.subheader("2. Wine Quality Dataset")
        st.markdown("Publically available, Wine Quality dataset is related to red and white wine variants. The dataset contains a total 6497 rows and 11 phsicochemical properties and 1 sensory characterstics(ranked from 0 to 10 scores) are used as input and out variables " )
        #st.image(r"/home/tess/Documents/python_projects/streamlit_app/images/wine_qty.png", width=None)
        st.subheader("Input Variables")
        st.write("1. Fixed acidity (g(tartaric acid)/L): Primary fixed acids found in wine are tartaric, sussinic, citric and malic acids")
        st.write("2. Volatile acidity (g(acetic acid)/L): Are the gaseaous acids present in wine")
        st.write("3. Citric acid (g/L): It's weak organic acid found in citrus fruits naturally.")
        st.write("4. Residual sugar (g/L): Amount of sugar left after fermentation.")
        st.write("5. Chlorides (g(sodium chloride)/L): Amount of salt present in wine.")
        st.write("6. Free surfur dioxide (mg/L): SO2 is used for prevention of wine from oxidation and microbial spoilage.")
        st.write("7. Total surfur dioxide (mg/L): ")
        st.write("8. Density (g/mL): The density of wine ranges from 0.933 g/ML to 0.995 g/mL")
        st.write("9. pH: The degree of wine acidity. It has a range between 2.9 4.2. Wine's chemical and biological properties are very dependent on pH value.")
        st.write("10. Sulphates (g(potassium sulphate)/L)")
        st.write("11. Alcohol (% vol: percent of alcohol present in wine.")
        st.subheader("Output variable")
        st.write("12. Quality( score between 0 and 10")
        st.subheader("Project Description")
        st.write("In this project, the model is trained to predict whether the wine is red or white solely from the atributes. As described below, the dataset contains 6479 input rows with 12 feature columns. In some columns, there are missing values. Fixed acidity, volatile acidity, citric acid, residual sugar, chloride, pH and sulphate columns have missing values 10, 8, 3, 2, 2, 9 and 4 respectively.")
        st.markdown("[Source: Iris Dataset project](https://medium.com/@sailajakonda2012/random-forest-classification-in-prediction-of-best-quality-wine-d0d7591a7c17)")
        st.write("Wine Quality Dataset")


        mycollection = db['wine quality']
        st.write(mycollection)
        all_records = mycollection.find()
        list_cursor = list(all_records)
        data = pd.DataFrame(list_cursor, columns=[
            'type', 'fixed acidity', 'volatile acidity',
            'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality' ])

        data['fixed acidity']  = pd.to_numeric(data['fixed acidity'], errors='coerce')
        data['volatile acidity']  = pd.to_numeric(data['volatile acidity'], errors='coerce')
        data['citric acid']  = pd.to_numeric(data['citric acid'], errors='coerce')
        data['residual sugar']  = pd.to_numeric(data['residual sugar'], errors='coerce')
        data['chlorides']  = pd.to_numeric(data['chlorides'], errors='coerce')
        data['free sulfur dioxide']  = pd.to_numeric(data['free sulfur dioxide'], errors='coerce')
        data['total sulfur dioxide']  = pd.to_numeric(data['total sulfur dioxide'], errors='coerce')
        data['density']  = pd.to_numeric(data['density'], errors='coerce')
        data['pH']  = pd.to_numeric(data['pH'], errors='coerce')
        data['sulphates']  = pd.to_numeric(data['sulphates'], errors='coerce')
        data['alcohol']  = pd.to_numeric(data['alcohol'], errors='coerce')
        data['quality']  = pd.to_numeric(data['quality'], errors='coerce')
        #data["variety"]=data["variety"].values.astype(str)
        #data = data.astype({"variety": str}, errors='raise') 
        
        #data['variety'].astype(str)   
        st.write(data)     
    #--------------------------------------------------------------
    elif dataset_name == "Iris":
        st.subheader("3. Iris Dataset")
        st.markdown("")
        #st.image(r"/home/tess/Documents/python_projects/streamlit_app/images/iris-dataset.png", width=None)
        st.markdown("[Source: Iris Dataset project](https://machinelearninghd.com/iris-dataset-uci-machine-learning-repository-project/)")
        st.write("Iris Dataset")

        mycollection = db['iris']
        st.write(mycollection)
        all_records = mycollection.find()
        list_cursor = list(all_records)
        data = pd.DataFrame(list_cursor, columns= ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])  
        data['SepalLengthCm']  = pd.to_numeric(data.SepalLengthCm, errors='coerce')      
        st.write(data)         
    
    elif dataset_name == "Breast cancer":
        st.subheader("4. Breast cancer Dataset")
        #st.image(r"/home/tess/Documents/python_projects/streamlit_app/images/breast_cancer.png", width=None)
        st.markdown("[Source: Cancer Research UK](https://www.cancerresearchuk.org/about-cancer/breast-cancer/stages-types-grades/tnm-staging)")
        st.write("Breast Cancer Dataset")

        mycollection = db['breast cancer']
        st.write(mycollection)
        all_records = mycollection.find()
        list_cursor = list(all_records)
        data = pd.DataFrame(list_cursor, columns=['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', "diagnosis"])
                
    #----------------------------------------------
    elif dataset_name == "Spam classifier":
        st.subheader("5. Spam Classifier Dataset")
        #st.image(r"/home/tess/Documents/python_projects/streamlit_app/images/spam_text.png", width=None)
        data_spam = pd.read_csv(r"/home/tess/Documents/python_projects/streamlit_app/data/spam.csv", encoding='ISO 8859-1')
        data_spam.rename({'v1': 'Label', 'v2': 'messages'}, axis=1, inplace=True)
        data = data_spam[['Label','messages']]
        data['label'] = data['Label'].apply(lambda x:1 if x=='spam' else 0)
        
    #-----------------------------------------------------------------------------------------------------
    
with eda:
    st.write(data)
    #data shape
    st.write("Shape of Dataset", data.shape)
    #Missing values in an object columnn
    st.write("Missing Data:", data.isnull().sum())
    data.drop_duplicates(inplace = True)
    st.write("Dataset Information:", data.describe())

#--------------------------------------------
with visualization:
    st.write("To explain the data, graphic representation such as Histograms, pairplot, pivot, correlation maps and 3D plots are used. In each visualization, the dataset is studied  and plotted accordingly. In some datasets, there are missing values. These missing values are not considered while plotting. Histograms and pivot graphs are used to understand the distribution of data. Correlation maps are developed to understand the relationship between features.")
    colors_list = ['#78C850', '#F08030',  '#6890F0',  '#A8B820',  '#F8D030', '#E0C068', '#C03028', '#F85888', '#98D8D8']
    if dataset_name =="Dolphins":
        fig, axs = plt.subplots(figsize=(16, 10))
        # Plot variable 1
        plt.subplot(2,2,1)
        ax1 = sns.histplot(data=data, x="dimension_1_mm", hue="variety", kde=True)
        #Plot variable 2
        plt.subplot(2,2,2)
        ax1 = sns.histplot(data=data, x="dimension_2_mm", hue="variety", kde=True)
        # Plot variable 3
        plt.subplot(2,2,3)
        ax1 = sns.histplot(data=data, x="dimension_3_mm", hue="variety", kde=True)
        # Plot variable 4
        plt.subplot(2,2,4)
        ax1 = sns.histplot(data=data, x="mass_g", hue="variety", kde=True)
        st.pyplot(fig)
        #--------------------------------------------
        #pair plot

        fig = plt.figure(figsize=(10,6))
        st.subheader("Features of Interest")
        st.markdown("This part relates to the selection of features (i.e. variables or columns) that would have a positive impact on the goodness of fit of our prediction model. For this, the correlation matrix is exploited as insights for selection. It is expected that most correlated features to the variety will have a positive impact on the goodness of fit of our model.")
        st.header("Data Visualization")
        
        # Mean distribution graphic visualiation, violin plot
        # pair plot
        st.subheader("Pair plot")
        plot = sns.pairplot(data, hue="variety")
        st.pyplot(plot)
        #----------------------------
        #Violin Plot
        #Violin Plot
        st.subheader("Violin Plot")
        st.markdown("Violin plot shows the concentration of data[for more details click this link](https://www.infinityinsight.com/blog/?p=357). ")        
        
        fig = plt.figure(figsize=(12,12))

        plt.subplot(2,2,1)
        df1 = data['dimension_1_mm']            
        sns.violinplot(data=df1, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df1)

        plt.subplot(2,2,2)
        df2 = data[['dimension_2_mm']]
        sns.violinplot(data=df2, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df2)
                
        plt.subplot(2,2,3)
        df3 = data[['dimension_3_mm']]
        sns.violinplot(data=df3, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df3)
               
        plt.subplot(2,2,4)
        df4 = data[['mass_g']]
        sns.violinplot(data=df4, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df4)
        st.pyplot(fig)
        #------------------------------
        #Correlation Matrix
        fig = plt.figure(figsize=(10,6))
        st.subheader("Dataset correlation diagram")
        sns.heatmap(data= data.select_dtypes(include=['float64', 'int64']).dropna().corr(), annot=True, fmt='.1g', linewidths=0.8)
        #plt.show()
        st.pyplot(fig)
        #------------------------------------
        # 3D scatter plot
        fig = plt.figure(figsize=(10,6))
        corr = data.select_dtypes(include=['float64', 'int64']).dropna().corr()
        st.write(corr)
        st.markdown("Based on the correlation matrix, features are given below by order of interest :")
        features_interest = data.select_dtypes(include=['float64', 'int64']).dropna().corr().iloc[1].sort_values(ascending=False)#[1:].index
        st.write(features_interest)
        st.markdown("3-D representation to show how the data may be correlated to multiple features.")
        scatter_3D = px.scatter_3d(data, x="dimension_2_mm", y="mass_g", z="variety", hover_name="dimension_3_mm", color="dimension_1_mm", width=1000, height=800)
        st.plotly_chart(scatter_3D)
#-------------------------------------------
    elif dataset_name=="Wine Quality":   

        st.subheader("Histogram Plot")
        fig, axs = plt.subplots(figsize=(12, 10))
        # Plot variable 1
        plt.subplot(3,4,1)
        ax1 = sns.histplot(data=data, x="fixed acidity", hue="type", kde=True)

        #Plot variable 2
        plt.subplot(3,4,2)
        ax1 = sns.histplot(data=data, x="volatile acidity", hue="type", kde=True)

        # Plot variable 3
        plt.subplot(3,4,3)
        ax1 = sns.histplot(data=data, x="citric acid", hue="type", kde=True)

        # Plot variable 4
        plt.subplot(3,4,4)
        ax1 = sns.histplot(data=data, x="residual sugar", hue="type", kde=True)
        
        #Plot variable 5
        plt.subplot(3,4,5)
        ax1=sns.histplot(data=data,x="free sulfur dioxide", hue="type", kde=True)

        #Plot variable 6
        plt.subplot(3,4,6)
        ax1=sns.histplot(data=data,x="free sulfur dioxide", hue="type", kde=True)

        #Plot variable 7
        plt.subplot(3,4,7)
        ax1=sns.histplot(data=data,x="total sulfur dioxide", hue="type", kde=True)

        #Plot variable 8
        plt.subplot(3,4,8)
        ax1=sns.histplot(data=data,x="density", hue="type", kde=True)

        #Plot variable 9
        plt.subplot(3,4,9)
        ax1=sns.histplot(data=data,x="pH", hue="type", kde=True)

        #Plot variable 10
        plt.subplot(3,4,10)
        ax1=sns.histplot(data=data,x="sulphates", hue="type", kde=True)

        #Plot variable 11
        plt.subplot(3,4,11)
        ax1=sns.histplot(data=data,x="alcohol", hue="type", kde=True)

        #Plot variable 9
        plt.subplot(3,4,12)
        ax1=sns.histplot(data=data,x="quality", hue="type", kde=True)
        st.pyplot(fig)
        #--------------------------------------------
        #Violin Plot
        #--------------------------------------------

        st.subheader("Violin Plot")
        st.markdown("Violin plot shows the concentration of data[for more details click this link](https://www.infinityinsight.com/blog/?p=357). ")        
       
        fig = plt.figure(figsize=(12,10))

        plt.subplot(3,4,1)
        df1 = data[['fixed acidity']]
        sns.violinplot(data=df1, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data=df1)

        plt.subplot(3,4,2)
        df2 = data[['volatile acidity']]
        sns.violinplot(data=df2, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df2)
                
        plt.subplot(3,4,3)
        df3 = data[['citric acid']]
        sns.violinplot(data=df3, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df3)
               
        plt.subplot(3,4,4)
        df4 = data[['residual sugar']]
        sns.violinplot(data=df4, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df4)
        
        plt.subplot(3,4,5)
        df5 = data[['chlorides']]
        sns.violinplot(data=df5, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df5)
                   
        plt.subplot(3,4,6)
        df6 = data[['free sulfur dioxide']]
        sns.violinplot(data=df6, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df6)

        plt.subplot(3,4,7)
        df7 = data[['total sulfur dioxide']]
        sns.violinplot(data=df7, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df7)

        plt.subplot(3,4,8)
        df8 = data[['density']]
        sns.violinplot(data=df8, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df8)

        plt.subplot(3,4,9)
        df9 = data[['pH']]
        sns.violinplot(data=df9, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df9)

        plt.subplot(3,4,10)
        df10 = data[['sulphates']]
        sns.violinplot(data=df10, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df10)

        plt.subplot(3,4,11)
        df11 = data[['alcohol']]
        sns.violinplot(data=df11, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df11)

        plt.subplot(3,4,12)
        df12 = data[['quality']]
        sns.violinplot(data=df12, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df12)       
        st.pyplot(fig)
        #-------------------------------------------
        #Correlation Matrix
        fig = plt.figure(figsize=(10,6))
        st.subheader("Dataset correlation diagram")
        sns.heatmap(data=data.dropna().corr(), annot=True, fmt='.1g', linewidths=0.8)
        plt.show()
        st.pyplot(fig)
        #------------------------------------
        # 3D scatter plot
        fig = plt.figure(figsize=(10,6))
        corr = data.dropna().corr()
        st.write(corr)
        #--------------------------------
        st.markdown("Based on the correlation matrix, features are given below by order of interest :")
        features_interest = data.select_dtypes(include=['float64', 'int64']).dropna().corr().iloc[1].sort_values(ascending=False)#[1:].index
        st.write(features_interest)

        st.markdown("3-D representation to show how the data may be correlated to multiple features.")
        scatter_3D = px.scatter_3d(data, x="fixed acidity", y="volatile acidity", z="type", hover_name="pH", color="citric acid", width=1000, height=800)
        st.plotly_chart(scatter_3D)

#----------------------------------------------
    elif dataset_name=="Iris":
        fig, axs = plt.subplots(figsize=(16, 10))
        # Plot variable 1
        plt.subplot(2,2,1)
        ax1 = sns.histplot(data=data, x="SepalLengthCm", hue="Species", kde=True)
        #Plot variable 2
        plt.subplot(2,2,2)
        ax1 = sns.histplot(data=data, x="SepalWidthCm", hue="Species", kde=True)
        # Plot variable 3
        plt.subplot(2,2,3)
        ax1 = sns.histplot(data=data, x="PetalLengthCm", hue="Species", kde=True)
        # Plot variable 4
        plt.subplot(2,2,4)
        ax1 = sns.histplot(data=data, x="PetalWidthCm", hue="Species", kde=True)
        st.pyplot(fig)

        #pair plot

        fig = plt.figure(figsize=(10,6))
        st.subheader("Features of Interest")
        st.markdown("This part relates to the selection of features (i.e. variables or columns) that would have a positive impact on the goodness of fit of our prediction model. For this, the correlation matrix is exploited as insights for selection. It is expected that most correlated features to the variety will have a positive impact on the goodness of fit of our model.")
        st.header("Data Visualization")
        
        # Mean distribution graphic visualiation, violin plot
        # pair plot
        st.subheader("Pair plot")
        plot = sns.pairplot(data, hue="Species")
        st.pyplot(plot)
        #----------------------------
        #Violin Plot
        st.subheader("Violin Plot")
        st.markdown("Violin plot shows the concentration of data[for more details click this link](https://www.infinityinsight.com/blog/?p=357). ")        
        
        fig = plt.figure(figsize=(12,12))

        plt.subplot(2,2,1)
        df1 = data['SepalLengthCm']            
        sns.violinplot(data=df1, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df1)

        plt.subplot(2,2,2)
        df2 = data[['SepalWidthCm']]
        sns.violinplot(data=df2, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df2)
                
        plt.subplot(2,2,3)
        df3 = data[['PetalLengthCm']]
        sns.violinplot(data=df3, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df3)
               
        plt.subplot(2,2,4)
        df4 = data[['PetalWidthCm']]
        sns.violinplot(data=df4, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df4)
        st.pyplot(fig)
        #------------------------------
        #Correlation Matrix
        fig = plt.figure(figsize=(10,6))
        st.subheader("Dataset correlation diagram")
        sns.heatmap(data= data.select_dtypes(include=['float64', 'int64']).dropna().corr(), annot=True, fmt='.1g', linewidths=0.8)
        plt.show()
        st.pyplot(fig)
        #------------------------------------
        # 3D scatter plot
        fig = plt.figure(figsize=(10,6))

        corr = data.select_dtypes(include=['float64', 'int64']).dropna().corr()
        st.write(corr)
        st.markdown("Based on the correlation matrix, features are given below by order of interest :")
        features_interest = data.select_dtypes(include=['float64', 'int64']).dropna().corr().iloc[1].sort_values(ascending=False)#[1:].index
        st.write(features_interest)

        #----------------------------
        st.markdown("3-D representation to show how the data may be correlated to multiple features.")
        scatter_3D = px.scatter_3d(data, x="SepalLengthCm", y="PetalLengthCm", z="Species", hover_name="SepalLengthCm", color="PetalWidthCm", width=1000, height=800)
        st.plotly_chart(scatter_3D)

    #----------------------------------------------
    elif dataset_name=="Breast cancer":
        fig, axs = plt.subplots(figsize=(16, 10))
        # Plot variable 1
        plt.subplot(3,2,1)
        ax1 = sns.histplot(data=data, x="mean_radius", hue="diagnosis", kde=True)
        #Plot variable 2
        plt.subplot(3,2,2)
        ax1 = sns.histplot(data=data, x="mean_texture", hue="diagnosis", kde=True)
        # Plot variable 3
        plt.subplot(3,2,3)
        ax1 = sns.histplot(data=data, x="mean_perimeter", hue="diagnosis", kde=True)
        # Plot variable 4
        plt.subplot(3,2,4)
        ax1 = sns.histplot(data=data, x="mean_area", hue="diagnosis", kde=True)

        # Plot variable 5
        plt.subplot(3,2,5)
        ax1 = sns.histplot(data=data, x="mean_smoothness", hue="diagnosis", kde=True)

        st.pyplot(fig)
    #-------------------------------
        #pair plot

        fig = plt.figure(figsize=(10,6))
        st.subheader("Features of Interest")
        st.markdown("This part relates to the selection of features (i.e. variables or columns) that would have a positive impact on the goodness of fit of our prediction model. For this, the correlation matrix is exploited as insights for selection. It is expected that most correlated features to the variety will have a positive impact on the goodness of fit of our model.")
        st.header("Data Visualization")
        
        # Mean distribution graphic visualiation, violin plot
        # pair plot
        st.subheader("Pair plot")
        plot = sns.pairplot(data, hue="diagnosis")
        st.pyplot(plot)
    #----------------------------
        #Violin Plot
        st.subheader("Violin Plot")
        st.markdown("Violin plot shows the concentration of data[for more details click this link](https://www.infinityinsight.com/blog/?p=357). ")        
    
        fig = plt.figure(figsize=(12,12))

        plt.subplot(3,2,1)
        df1 = data['mean_radius']            
        sns.violinplot(data=df1, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df1)

        plt.subplot(3,2,2)
        df2 = data[['mean_texture']]
        sns.violinplot(data=df2, orient="v", scale="count", palette=colors_list, split=True)
        sns.stripplot(data = df2)
                
        plt.subplot(3,2,3)
        df3 = data[['mean_perimeter']]
        sns.violinplot(data=df3, orient="v", scale="count", palette=colors_list, split=True, edgecolor = 'blue')
        sns.stripplot(data = df3)
        
               
        plt.subplot(3,2,4)
        df4 = data[['mean_area']]
        sns.violinplot(data=df4, orient="v", scale="count", palette=colors_list, split=True, edgecolor = 'blue')
        sns.stripplot(data = df4)

        plt.subplot(3,2,5)
        df5 = data[['mean_smoothness']]
        sns.violinplot(data=df4, orient="v", scale="count", palette=colors_list, split=True, edgecolor = 'blue')
        sns.stripplot(data = df5)

        st.pyplot(fig)
        #------------------------------

        #Correlation Matrix
        fig = plt.figure(figsize=(10,6))
        st.subheader("Dataset correlation diagram")
        sns.heatmap(data= data.dropna().corr(), annot=True, fmt='.1g', linewidths=0.8, edgecolor = 'blue')
        plt.show()
        st.pyplot(fig)
        #------------------------------------
        # 3D scatter plot
        fig = plt.figure(figsize=(10,6))
        corr = data.dropna().corr()
        st.write(corr)
        st.markdown("Based on the correlation matrix, features are given below by order of interest :")
        features_interest = data.select_dtypes(include=['float64', 'int64']).dropna().corr().iloc[1].sort_values(ascending=False)#[1:].index
        st.write(features_interest)
        st.markdown("3-D representation to show how the data may be correlated to multiple features.")
        scatter_3D = px.scatter_3d(data, x="mean_radius", y="mean_texture", z="diagnosis", hover_name="mean_perimeter", color="mean_area", width=1000, height=800)
        st.plotly_chart(scatter_3D)
    #---------------------------------------

    elif dataset_name == "Spam classifier":
        #----------------------------
        # define length size, length value selected based on mximum spam size, exemple: len<200

        data['length'] = data['messages'].str.len()
        df2 = data[data['length']<200]

        #Histogram plot
        st.subheader("Histogram plot")
        fig = plt.figure(figsize=(10,8))
        ax1 = sns.histplot(data=df2, x="length", hue= "Label", kde=True, edgecolor = 'blue')
        st.pyplot(fig)
    #-------------------------------
        #Violin plot

        st.subheader("Mean distribution grapgh, violin plot")
        fig = plt.figure(figsize=(12,8))

        ax = sns.violinplot(x =df2['Label'], y =df2['length'], data=df2, inner=None, color=".8")

        ax = sns.stripplot(x =df2['Label'], y =df2['length'], data=df2)

        ax.set_title('Messages with length < 00')
        st.pyplot(fig)
    #---------------------------------
with Preprocessing:
    st.header("Preprocessing")
    
    if dataset_name=="Dolphins":
        #Target dataset, y
        st.subheader("Preprocessing for Target, y")
        y = data['variety']
        lb_encod = LabelEncoder()
        y = lb_encod.fit_transform(y)
        st.write("Target", y.shape)
        #Features
        st.subheader('Features Dataset')
        X = data.drop(columns='variety')

        st.write("Features", X.shape)

        #columns
        columns = ['variety', 'area', 'dimension_1_mm', 'dimension_2_mm', 'dimension_3_mm', 'mass_g', 'sex']

        #selection of variables
        #selection of categories variable
        column_cat = X.select_dtypes(include=['object']).columns
        #column_cat
        
        transfo_cat = Pipeline(steps=[('imputation', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore', sparse = False))])
        

        #selection of numeric variables        
        column_num = X.select_dtypes(exclude=['object']).columns
        #column_num
        transfo_num = Pipeline(steps=[('imputation', SimpleImputer(strategy='median')),('scaling', MaxAbsScaler())])
       
    #----------------------------------------
    elif dataset_name == "Wine Quality":
        #Target
        st.subheader("Preprocessing for Target, y")
        y = data['type']
        lb_encod = LabelEncoder()
        y = lb_encod.fit_transform(y)
        st.write("Target", y)
        st.write("Target", y.shape)
        #Features
        st.subheader('Preprocessing for Features Dataset variables')
        X = data.drop(columns='type')
        st.write("Features", X.shape)

        #columns
        columns = [
            'type', 'fixed acidity', 'volatile acidity',
            'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur diioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality' ]

        #selection of variables
        #selection of categories variable
        column_cat = X.select_dtypes(include=['object']).columns

        #column category transformation
        transfo_cat = Pipeline(steps=[('imputation', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore', sparse = False))])
        
        
        #selection of numeric variables        
        column_num = X.select_dtypes(exclude=['object']).columns
        #column_num
        transfo_num = Pipeline(steps=[('imputation', SimpleImputer(strategy='median')),('scaling', MinMaxScaler())])
        
    #-----------------------------------
    elif dataset_name=="Iris":
        st.subheader("Preprocessing for Target, y")
        y=data['Species']
        
        lb_encod = LabelEncoder()
        y = lb_encod.fit_transform(y)
        st.write("Target", y)
        st.write("Target", y.shape)

        #Features
        st.subheader('Preprocessing of Features Dataset Variables')
        X = data.drop(columns='Species')

        st.write("Features", X.shape)

        #columns
        columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

        #selection of variables
        #selection of categories variable
        column_cat = X.select_dtypes(include=['object']).columns

        #column transformation

        #column_cat
        
        transfo_cat = Pipeline(steps=[('imputation', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore', sparse = False))])
        
        #selection of numeric variables        
        column_num = X.select_dtypes(exclude=['object']).columns
        #column_num
        transfo_num = Pipeline(steps=[('imputation', SimpleImputer(strategy='median')),('scaling', MinMaxScaler())])
        
#-------------------------------------------------
    elif dataset_name=="Breast cancer":
        st.subheader("Preprocessing for Target, y")
        y=data['diagnosis']
        st.write("Target", y)
        st.write("Target", y.shape)

        #Features
        st.subheader('Preprocessing of Features Dataset variables')
        X = data.drop(columns='diagnosis')
        st.write("Features", X.shape)

        #columns
        columns = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', "diagnosis"]

        #selection of variables
        #selection of categories variable
        column_cat = X.select_dtypes(include=['object']).columns

        #column category transformation
        transfo_cat = Pipeline(steps=[('imputation', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore', sparse = False))])
        
        
        #selection of numeric variables        
        column_num = X.select_dtypes(exclude=['object']).columns
        #column_num
        transfo_num = Pipeline(steps=[('imputation', SimpleImputer(strategy='median')),('scaling', MinMaxScaler())])
            
    elif dataset_name=="Spam classifier":
    #--------------------------
        #Features
        st.subheader('Features Engineering')
        #lowerization
        data = data.applymap(lambda s: s.lower() if type(s) == str else s)
        #data.head()
        #Features extraction
        st.subheader("Features Creation using Regex")

        features_regex={'Features':['tel_num_3', 'tel_num_5', 'tel_num_10', 'tel_num_11', 'money_pound', 'money_dollar', 'special_char', 'emails', 'dates1', 'dates2'],
        'Regex_char':['\d{3}', '\d{5}', '\d{3} \d{3} \d{4}', '\d{11}', '£\d+', '\$\d+', '&lt;#&gt;', '[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,4}', '\d+\s\w+\s\d+', '\d+/\d+/\d']}
        data_features = pd.DataFrame(features_regex)
                
        st.table(data_features)
        #phone numbers, with 3 digits

        #Digits 3, commonly used for text messaging
        tel_num_3 = data['messages'].str.contains("\d{3}")
        data['tel_num_3'] = 1*tel_num_3

        #Digits 5
        tel_num_5 = data['messages'].str.contains("\d{5}")
        data['tel_num_5'] = 1*tel_num_5

        #Digits 10
        tel_num_10 = data['messages'].str.contains('\d{3} \d{3} \d{4}')
        data['tel_num_10'] = 1*tel_num_10

        #Digits 11
        tel_num_11 = data['messages'].str.contains(('\d{11}'))
        data['tel_num_11'] = 1*tel_num_11

        #Asking for money
        #asking for money, "[^a-z].*£(\d+)"

        money_pound = data['messages'].str.contains("£\d+")
        data['money_pound'] = 1*money_pound

        money_dollar = data['messages'].str.contains("\$\d+")
        data['money_dollar'] = 1*money_dollar

        #Mixed Special characters commonly visible
        special_char = data['messages'].str.contains('&lt;#&gt;')
        data['special_char'] = 1*special_char

        #emails [\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,4}
        emails = data['messages'].str.contains("[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,4}")
        data['emails'] = 1*emails

        #Dates (\d+\s\w+\s\d+)
        dates1 = data['messages'].str.contains("\d+\s\w+\s\d+")
        data['dates1'] = 1*dates1

        #Dates "\d+/\d+/\d"
        dates2 = data['messages'].str.contains("\d+/\d+/\d")
        data['dates2'] = 1*dates2

        st.subheader("Spam Text Processing")

        #Spam messages
        spams = data.loc[data['Label']=='spam', :]
        #st.write("Spam messages and their features:", spams.head())

        #Remove punctuations

        #Download the stopwords package

        nltk.download('stopwords')

        #download punctuation package

        nltk.download('punkt')

        #download wordnet package

        nltk.download('wordnet')

        #punctuations
        punctuations = string.punctuation

        #defining the function to remove punctuation

        def remove_punctuation(text):
            punctuationfree="".join([i for i in text if i not in punctuations])
            return punctuationfree

        #storing the puntuation free text

        clean_spam1 = spams['messages'].apply(lambda x:remove_punctuation(x))
        #spams['Text_nonpunc']

        #Removing stop words from the spam text
        #defining the function to remove stopwords from tokenized text

        def remove_stopwords(txt):
            
            stoplist = nltk.corpus.stopwords.words('english')

            output =' '.join([word for word in txt.split() if word not in stoplist])
            return output

        ##applying the stopword function

        clean_spam1 = clean_spam1.apply(lambda x:remove_stopwords(x))
        #spams['clean_text'].head()

        #split words to columns

        splitwords = [nltk.word_tokenize(str(sentence)) for sentence in clean_spam1]
        #splitwords

        st.subheader("Word Cloud Vizualisation")
        
        #Libraries
        #--------------------------

        from io import StringIO
        import matplotlib as mpl 
        from matplotlib.pyplot import figure
        from matplotlib import rcParams
        from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
        #--------------------------------

        #bank of words, bows

        bows = [nltk.word_tokenize(str(word)) for word in clean_spam1]
        #bows[0:2], #example for row[index = ['':'']
        
        #word counts
        counts = [len(words) for words in bows]

        #char_length
        char_length =clean_spam1.apply(len)

        #word cloud
        st.subheader("Word Cloud visualization")

        df = data
        column = clean_spam1
        numWords = counts

        def wordCloudFunction(df,column,numWords):
            topic_words = [ z.lower() for y in
                            [ x.split() for x in clean_spam1 if isinstance(x, str)]
                            for z in y]
            
            word_count_dict = dict(Counter(topic_words))
            
            popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
            
            popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]
            word_string=str(popular_words_nonstop)
            
            wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', max_words=numWords, width=1200,height=700,
                                ).generate(word_string)
            plt.clf()
            plt.imshow(wordcloud)
            plt.axis('off')
            plt.show()

            #spam word cloud, 1000 words

        fig = plt.figure(figsize=(15,15))
        wordCloudFunction(clean_spam1,'title',1000)
        st.pyplot(fig)

        st.subheader("Top Most Frequent spam words ")

        from collections import Counter

        df = data
        column =clean_spam1

        def wordBarGraphFunction(df,column,title):
            topic_words = [ z.lower() for y in
                        [x.split() for x in clean_spam1 if isinstance(x, str)]
                            for z in y]
            
            word_count_dict = dict(Counter(topic_words))
            
            popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
            
            popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]
            
            plt.barh(range(50), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:50])]) #30 top most common words
            plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[0:50]))
            plt.title(title)
            plt.show()
        fig = plt.figure(figsize=(10,10))
        wordBarGraphFunction(clean_spam1,'title',"Top most frequent spam words")
        st.pyplot(fig)

        #creating additional features from the most frequent words
        st.subheader("Features Creation: Top 20 Most Frequent word")
        data['call'] = data['messages'].apply(lambda x: 1 if r"call" in x else 0 )
        data['free'] = data['messages'].apply(lambda x: 1 if r"free" in x else 0 )
        data['txt'] = data['messages'].apply(lambda x: 1 if r"txt" in x else 0 )
        data['text'] = data['messages'].apply(lambda x: 1 if r"text" in x else 0 )
        data['urgent'] = data['messages'].apply(lambda x: 1 if r"urgent" in x else 0 )
        data['new'] = data['messages'].apply(lambda x: 1 if r"new" in x else 0 )
        data['reply'] = data['messages'].apply(lambda x: 1 if r"reply" in x else 0 )
        data['prize'] = data['messages'].apply(lambda x: 1 if r"prize" in x else 0 )
        data['get'] = data['messages'].apply(lambda x: 1 if r"get" in x else 0 )
        data['claim'] = data['messages'].apply(lambda x: 1 if r"claim" in x else 0 )
        data['call'] = data['messages'].apply(lambda x: 1 if r"call" in x else 0 )
        data['contact'] = data['messages'].apply(lambda x: 1 if r"contact" in x else 0 )
        data['win'] = data['messages'].apply(lambda x: 1 if r"win" in x else 0 )
        data['cash'] = data['messages'].apply(lambda x: 1 if r"cash" in x else 0)
        data['phone'] = data['messages'].apply(lambda x: 1 if r"phone" in x else 0 )
        data['service'] = data['messages'].apply(lambda x: 1 if r"service" in x else 0 )
        data['guaranteed'] = data['messages'].apply(lambda x: 1 if r"guaranteed" in x else 0 )
        data['awarded'] = data['messages'].apply(lambda x: 1 if r"awarded" in x else 0 )
        
        #Preparation of Input Data
        st.subheader("Preparation of Input Data")
        st.write("Creation of a balanced Data by filtering the Spam messages")
        # creating new dataframe as df_ham , df_spam
        st.write("Spam dataset")

        df_spam = data.loc[data['Label']=='spam', :]
        st.write("spam Dataset Shape:", df_spam.shape)

        #Filtering Ham dataset
        st.write("Ham Dataset")
        df_ham = data.loc[data['Label']=='ham', :]
        st.write("Ham Dataset Shape:", df_ham.shape)

        #Creation of balanced data
        st.subheader("Balancing Data, Downsampling")
        st.markdown("As there could be bias in the dataset of the given data sample, due to major difference, downsampling technique is applied. Downsampling is performed to reduce the number of samples that is expected to have a bias class. This arises mainly because of high differences between the sample classes. Hence, downsampling is a technique where the majority class is downsampled to match minority class.")
        st.markdown("It is calculated by $ minority(spam) values/(majority(ham) values*100% $")
        st.markdown("Percentage of the target data, spam")
        # Percentage of data to be balanced- spam by downsampling

        #ham_data and spam_data shape[] are used for calculating Percentage
        
        Perc_spam = (df_spam.shape[0]/df_ham.shape[0])*100 #Perc_spam = Percentage of spam, in %
        st.write('Spam Percentage:', Perc_spam)

        # downsampling ham dataset - Equal to the '%' of spam
        df_ham_downsampled = df_ham.sample(df_spam.shape[0])

        # concating both dataset - df_spam and df_ham_balanced to create df_balanced dataset
        #50-50

        df_perc = pd.concat([df_spam , df_ham_downsampled]) #Balanced Data df_balanced  = df
        
        
        #dataset after downsampling. This data is used as an input for model training
        st.write("Dataset with created features(50%\ spam, 50%\Hamp):", df_perc.head()) 
#-------------------------------
        st.subheader("Correlation Diagram of spam features")
        #Correlation Matrix
        fig = plt.figure(figsize=(20,15))
        sns.heatmap(data= df_perc.dropna().corr(), annot=True, fmt='.1g', linewidths=0.1, annot_kws={"size":12}, edgecolor = 'blue')
        #plt.show()
        st.pyplot(fig)
#---------------------------------
        st.subheader("Features with most Importance")

        corr = df_perc.select_dtypes(include=['float64', 'int64']).dropna().corr()
        st.write(corr)
        st.markdown("Based on the correlation matrix, features are given below by order of interest :")
        features_interest = df_perc.select_dtypes(include=['float64', 'int64']).dropna().corr().iloc[1].sort_values(ascending=False)#[1:].index
        #sns.barplot(y=all_features, x=importance)

        st.write(features_interest)
#---------------------------------
        #Target dateset, y
        st.subheader("Preprocessing for Target, y")
        y=df_perc['Label']
        #label encoder
        lb_encod = LabelEncoder()
        y = lb_encod.fit_transform(y)
        st.write("Target", y)

        #Transformation
        #Features
        st.subheader('Preprocessing of Features Dataset variables')
        X = df_perc.drop(columns=['label', 'length', 'messages'])
        st.write("Features", X.shape)

        #3D Scatter plot
        st.subheader("3D Scatter Plot")

        fig = plt.figure(figsize=(10,6))
        st.markdown("3-D representation to show how the data may be correlated to multiple features.")
        scatter_3D = px.scatter_3d(df_perc, x="call", y="tel_num_3", z="label", hover_name="money_pound", color="txt", width=1000, height=800)
        st.plotly_chart(scatter_3D)

        #columns
        columns = ['label','length', 'tel_num_3', 'tel_num_5', 'tel_num_10', 'tel_num_11', 'money_pound', 'money_dollar', 'special_char', 'emails', 'dates1', 'dates2', 'call', 'free', 'txt', 'text', 'urgent', 'new', 'reply', 'prize', 'get', 'claim', 'contact', 'win', 'cash', 'phone', 'service', 'guaranteed', 'awarded']

        #selection of variables
        #selection of categories variable
        column_cat = X.select_dtypes(include=['object']).columns

        #column category transformation
                
        transfo_cat = Pipeline(steps=[('imputation', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore', sparse = False))])
        #col_cat = ColumnTransformer(['transfo_cat', ])
        
        #selection of numeric variables        
        column_num = X.select_dtypes(exclude=['object']).columns
        #column_num
        transfo_num = Pipeline(steps=[('imputation', SimpleImputer(strategy='median')),('scaling', MinMaxScaler())])
#---------------------------$
with transformation:
    #Encoder
    preparation = ColumnTransformer(transformers=[('data_cat', transfo_cat , column_cat),('data_num', transfo_num , column_num)])
    st.write(preparation)
#---------------------------
with Train_test_split:
    st.subheader("Data Initiation: Train test split")
    
    # features_test_data = 20%(X); target_test_data = 20%(y)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 42, stratify=y)

    # Train-Test dataset
    st.subheader("Train_test Dataset")

    st.write("X_train:", X_train.shape)
    st.write("X_test:", X_test.shape)
    st.write("y_train:", y_train.shape)
    st.write("X_train:", y_test.shape)
#---------------------------------------

with Parameters:
    params = dict()
    st.sidebar.subheader("Hyperparameters")
    st.subheader("Description of the Most Important parameters related terminologies")
    st.write("**n_estimators**: the number of decision trees in the model.")
    st.write("**Criterion**: It allows to select loss function used to determine model outcomes. Loss functions include mean squared error(**MSE**) and mean absolute error(**MAE**).")
    st.write("**Max-depth**: It sets the maximum possible depth of each decision tree.")
    st.write("**Max_features**: the maximum number of features the model will consider when determining a split.")
    st.write("**Bootstrap**: It allows to calculate standard errors, construct confidence intervals, and perform hypothesis testing for numerous types of sample statistics. It could be $True$ or $False$.")
    st.write("**Max_samples**: It assumes bootstrapping is set to True, if not, it's not applied. In the case of True, this value sets the largest size of each sample for each tree.")
    st.write("**n_jobs**: Number of processes you wish to run in parallel for this task. If it is $-1$, it will use all available processors.")
    st.write("**cv**: Determines the cross-validation splitting strategy. Possible inputs - None, integer(to specify the number of folds in a stratified KFold, which is applied in this project), CV splitter or an an iterable yielding(train, test) splits as arrays of indices.")
    st.write("**Estimator**: Pass the model instances for which you want to check the hyperparameters.")
    st.write("**Verbose**: it can be set to 1 to get the detailed print out while you fit the data to GridsearchCV. It controls the verbosity: the higher, the more messages.$>1$: the computation time for each fold and parameter candidate is displayed. $>2$: the score is also desplayed; $>3$: the fold and candidate parameter indexes are also desplayed together with the starting time of the computation")
    st.write("**Learning Rate**: Learning rate shrinks the distribution of each tree by learning rate.")
    st.write("**Sub sample**: The fraction of samples to be used for fitting the individual base learners.")
    def add_parameters(classifier_name):
        if classifier_name=="KNN":
            K = st.sidebar.slider("K", 1,15,(2,10),1)
            params["K"]= K
            n_jobs = st.sidebar.select_slider("Number of jobs to run in parallel (n_jobs)", options=[1,-1])
            params["n_jobs"]= n_jobs
            verbose = st.sidebar.slider("Verbose value", 1,2,3,4)
            params['verbose']=verbose            
               
        elif classifier_name == "SVM":
                #C = st.sidebar.slider("C", 0.01,10.0)
            C = st.sidebar.slider("C (Regularization parameter)", 0.01, 10.0, (0.01, 10.0), 2.0)
            params["C"]= C
        
            #kernel = st.sidebar.radio(label="Kernel", options=["rbf", "linear"], key="kernel")
            #params["kernel"]= kernel
            gamma = st.sidebar.slider("Gamma (Kernal coefficient", 0.0001,1.0, (0.0001,1.0), 0.05)
            params["gamma"]= gamma

            n_jobs = st.sidebar.select_slider("Number of jobs to run in parallel (n_jobs)", options=[1,-1])
            params["n_jobs"]= n_jobs
            verbose = st.sidebar.slider("Verbose value", 1,2,3,4)
            params['verbose']=verbose
            
        elif classifier_name == "Decision Tree Classifier":
            max_depth = st.sidebar.slider("Maximum depth", 3,10, (3,10), 1)
            params["max_depth"] = max_depth
            criterion = st.sidebar.select_slider("criterion", options=("gini", "entropy"))
            params["criterion"]= criterion
            #min samples leafs
            min_samples_leaf = st.sidebar.slider("Minimum samples leaf", 1,21, (1,21), 1)
            params["min_samples_leaf"] = min_samples_leaf

            cv=st.sidebar.slider("Number of Cross validation split", 2, 10)
            params["cross_validation"]= cv

            #min samples split
            min_samples_split = st.sidebar.slider("Minimum samples split", 2,11, (2,11), 2) #array
            params["min_samples_split"] = min_samples_split
            

            n_jobs = st.sidebar.select_slider("Number of jobs to run in parallel (n_jobs)", options=[1,-1])
            params["n_jobs"]= n_jobs
            verbose = st.sidebar.slider("Verbose value", 1,2,3,4)
            params['verbose']=verbose
            
        elif classifier_name == "Gradient Boosting Classifier":
            #training_rate=st.sidebar.slider("Training Rate", 0.01,0.30, (0.01,0.30), 0.05)
            #params["training_rate"]=training_rate
            n_estimators = n_estimators = st.sidebar.slider("Number of estimator", 10, 100, (10,100), 10)
            params["n_estimators"] = n_estimators

            max_depth = st.sidebar.slider("Maximum depth", 3,10, (3,10), 1)
            params["max_depth"] = max_depth
            #min samples leafs
            min_samples_leaf = st.sidebar.slider("Minimum samples leaf", 1,10, (1,10), 1)
            params["min_samples_leaf"] = min_samples_leaf

            criterion = st.sidebar.select_slider("criterion", options=("friedman_mse", "mae")) #mse=mean squared error, mae=mean absolute error
            params["criterion"]= criterion

            loss = st.sidebar.multiselect("Loss", ["deviance","exponential"], ["deviance"])
            params["loss"] = loss

            cv=st.sidebar.slider("Number of Cross validation split", 2, 10)
            params["cross_validation"]= cv

            #features
            #max_features =st.sidebar.multiselect("Max Features",["auto", 0.4], ["auto"])
            #params["max_features"]= max_features

            #random_state = st.sidebar.slider("Seed number (random_state)", 0,1000,42,1)
            #params["random_state"]= random_state

            #min samples split
            min_samples_split = st.sidebar.slider("Minimum samples split", 2,11, (2,11), 2) #array
            params["min_samples_split"] = min_samples_split

            n_jobs = st.sidebar.select_slider("Number of jobs to run in parallel (n_jobs)", options=[1,-1])
            params["n_jobs"]= n_jobs
            verbose = st.sidebar.slider("Verbose value", 1,2,3,4)
            params['verbose']=verbose

            
    #----------------------------------------
            #Random Forest classifier
        elif classifier_name == "Random Forest Classifier":

            n_estimators = st.sidebar.slider("Number of estimator", 0, 500, (10,50), 50)
            n_estimators_step = st.sidebar.number_input("Steps", 10)
            n_estimators_range = np.arange(n_estimators[0], n_estimators[1]+n_estimators_step, n_estimators_step)
            params["n_estimators_range"]= n_estimators_range
            #---------------------------------
            #n_estimators = st.sidebar.slider("The number of trees in the forest", 0, 100, key="n_estimatorss")
            max_depth = st.sidebar.slider("Maximum depth", 5, 15, (5,8), 2)
            max_depth_step=st.sidebar.number_input("Step size for max depht",1,3)
            max_depth_range =np.arange(max_depth[0],max_depth[1]+max_depth_step, max_depth_step)
            params["max_depth_range"]= max_depth_range

            #max_depth = st.sidebar.slider("The maximum depth of tree", 1, 15)
            
            max_features =st.sidebar.multiselect("Max Features (You can select multiple options)",["auto", "sqrt", "log2"],["auto"])
            params["max_features"]= max_features
            st.sidebar.write("---")
            criterion = st.sidebar.select_slider("criterion", options=("gini", "entropy"))
            params["criterion"]= criterion
            

            cv=st.sidebar.slider("Number of Cross validation split", 2, 10)
            params["cross_validation"]= cv

            st.sidebar.subheader("Other Parameters")
            random_state = st.sidebar.slider("Seed number (random_state)", 0, 1000, 42, 1)
            params["random_state"]= random_state

            bootstrap = st.sidebar.select_slider("Bootstrap", options=[True, False])
            params["bootstrap"]= bootstrap

            n_jobs = st.sidebar.select_slider("Number of jobs to run in parallel (n_jobs)", options=[1,-1])
            params["n_jobs"]= n_jobs
            verbose = st.sidebar.slider("Verbose value", 1,2,3,4)
            params['verbose']=verbose
        elif classifier_name == "Random Forest Regressor":
            #n_estimators
            n_estimators = st.sidebar.slider("Number of estimator", 0, 500, (10,50), 50)
            n_estimators_step = st.sidebar.number_input("Steps", 10)
            n_estimators_range = np.arange(n_estimators[0], n_estimators[1]+n_estimators_step, n_estimators_step)
            params["n_estimators_range"]= n_estimators_range

            #max depth

            max_depth = st.sidebar.slider("Maximum depth", 5, 15, (5,8), 2)
            max_depth_step=st.sidebar.number_input("Step size for max depht",1,3)
            max_depth_range =np.arange(max_depth[0],max_depth[1]+max_depth_step, max_depth_step)
            params["max_depth_range"]= max_depth_range

            #features
            max_features =st.sidebar.multiselect("Max Features (You can select multiple options)",["auto", "log2","sqrt"],["sqrt"])
            params["max_features"]= max_features

            criterion = st.sidebar.select_slider("criterion", options=("mse", "mae")) #mse=mean squared error, mae=mean absolute error
            params["criterion"]= criterion

            #min samples split
            min_samples_split = st.sidebar.slider("Minimum samples split", 2,8, (2,8), 2) #array
            params["min_samples_split"] = min_samples_split

            #min samples leafs
            min_samples_leaf = st.sidebar.slider("Minimum samples leaf", 1,4, (1,4), 1)
            params["min_samples_leaf"] = min_samples_leaf

            #cross validation

            cv=st.sidebar.slider("Number of Cross validation split", 2, 10)
            params["cross_validation"]= cv

            #random state

            random_state = st.sidebar.slider("Seed number (random_state)", 0, 1000, 42, 1)
            params["random_state"]= random_state

            #boot strap
            bootstrap=st.sidebar.select_slider("Bootstrap", options=[True, False])
            params["bootstrap"]= bootstrap

            n_jobs = st.sidebar.select_slider("Number of jobs to run in parallel (n_jobs)", options=[1,-1])
            params["n_jobs"]= n_jobs
            verbose = st.sidebar.slider("Verbose value", 1,2,3,4)
            params['verbose']=verbose

        return params
    params=add_parameters(classifier_name)
    #st.write("Parameters:", params)

            #"References
            # 1. https://www.analyticsvidhya.com/blog/2021/12/ml-hyperparameter-optimization-app-using-streamlit/"
            # 2. https://www.analyticsvidhya.com/blog/2021/05/a-brief-introduction-to-building-interactive-ml-webapps-with-streamlit/
with model:
    if classifier_name == "KNN":
        st.subheader("KNeighbors Classifier Model")
        model = KNeighborsClassifier()
        param_grid = dict(
            n_neighbors=params["K"])

        st.subheader("Grid Search")    
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=params["n_jobs"], verbose=params['verbose'])
        st.write("Grid Search:", grid)
        
    elif classifier_name == "SVM":
        st.subheader("Support Vector Machine Model")
        model = SVC()
        param_grid=dict(
            C=params["C"],
            gamma=params["gamma"])
        
        st.subheader("Grid Search")    
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=params["n_jobs"], verbose=params['verbose'])
        st.write("Grid Search:", grid)
       
    elif classifier_name == "Decision Tree Classifier":
        st.subheader("Decision Tree Classifier Model")
        
        model = DecisionTreeClassifier(criterion=params["criterion"])
        param_grid = dict(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"])

        st.subheader("Grid Search")    
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=params["cross_validation"], n_jobs=params["n_jobs"], verbose=params['verbose'])
        st.write("Grid Search:", grid)
        
    elif classifier_name == "Gradient Boosting Classifier":
        st.subheader("Gradient Boosting Classifier Model")
        model = GradientBoostingClassifier(criterion=params["criterion"])
        param_grid = dict(
            max_depth=params["max_depth"],
            loss = params["loss"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"])
        
        st.subheader("Grid Search")    
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=params["cross_validation"], n_jobs=params["n_jobs"], verbose=params['verbose'])
        st.write("Grid Search:", grid)
        
    elif classifier_name == "Random Forest Classifier":
        st.subheader("Random Forest Classifier Model")
    #model
        model = RandomForestClassifier(random_state=params["random_state"], bootstrap=params["bootstrap"])
        param_grid = dict(
            max_features=params["max_features"],
            n_estimators=params["n_estimators_range"],
            max_depth=params["max_depth_range"])

        st.subheader("Grid Search")    
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=params["cross_validation"], n_jobs=params["n_jobs"], verbose=params['verbose'])
        st.write("Grid Search:", grid)         
    elif classifier_name == "Random Forest Regressor":
        st.subheader("Random Forest Regressor Model")
        model = RandomForestRegressor(random_state=params["random_state"], bootstrap=params["bootstrap"], criterion=params["criterion"])
        param_grid = dict(
            max_features=params["max_features"],
            n_estimators=params["n_estimators_range"],
            max_depth=params["max_depth_range"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"])
    
        st.subheader("Grid Search")    
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=params["cross_validation"], n_jobs=params["n_jobs"], verbose=params['verbose'])
        st.write("Grid Search:", grid)
        
#-------------------------------------------
with pipeline:
    pipe = Pipeline(steps=[('preprocessor', ColumnTransformer(remainder='passthrough', transformers=[('data_cat', transfo_cat , column_cat),('data_num', transfo_num , column_num)])), ('classifier', grid)])
    st.write(pipe)
    
#--------------------------------------
#from mlxtend.plotting import plot_confusion_matrix
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import mean_squared_error
with model_training:
    if classifier_name == "Random Forest Regressor":
        #st.sidebar("Models")
        st.markdown("Make prediction for the pre-trained model")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        st.write("Prediction", y_pred)
        
        #Evaluation
        st.subheader("Calculating Error")
        #y_pred = pipe.predict(X_test)
        
        st.subheader("Mean Square Error, MSE")
        st.markdown("MSE is the most common loss function. It's defined as Mean or average of the square of the difference between actual and estimated values. MSE is used to check how close predictions are to actual values and hence it ensures the trained model to have no outlier predictions with significant errors. Its equation is as given below. For further readings, [please click here](https://www.mygreatlearning.com/blog/mean-square-error-explained/).")
        st.markdown(r"""$MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-y_i^{-})²$""")
        st.write("For the model applied, the value obtained is:")

        rnd_MSE = mean_squared_error(y_test, y_pred)
        st.write("Mean Squared Error, MSE:", rnd_MSE)
        #Calculating the Root Mean Squared Error
        
        st.subheader("Root Mean Square Errot, RMSE")
        st.markdown("[RMSE](https://www.sciencedirect.com/topics/engineering/root-mean-squared-error) is the standatd deviation of the residual(prediction errors. It measures how far the residuals are from the regresssion line data. In short, it shows how concentrated the data is around the line of best fit. When standarized observations($O_i$) are predictions($S_i$) are used as RMSE inputs, there is a direct relationship with the correlationn coefficient. For example, if the correlation is 1, the RMSE will be 0, because all of the points lie on the regression line and hence there is no error. ")
        st.markdown(r"""$RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (S_i-O_i)^{2}}$""")
        rnd_RMSE = np.sqrt(rnd_MSE)
        st.write("Root Mean Squared Error, RMSE:", rnd_RMSE)

        st.subheader("Accuracy, Precision and Recall")
        accuracy = pipe.score(X_test, y_test)
        st.write("Accuracy:", accuracy.round(2))
                
    else: 
            #st.sidebar("Models")
        st.markdown("Make prediction for the pre-trained model")
        pipe.fit(X_train, y_train)
        st.subheader("Model prediction")
        y_pred = pipe.predict(X_test)
        st.write("Prediction:", y_pred)
    
        #Matric confusion
        st.subheader("Matrix Confusion")
        fig = plt.figure()
        cm = confusion_matrix(y_test, y_pred)
        st.write("Martrix Confusion:", cm)
        sns.heatmap(cm, annot=True, cmap=plt.cm.YlOrBr)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('classification_1.png', dpi=300)
        st.pyplot(fig)

        #Evaluation
        st.subheader("Calculating Error")
        #y_pred = pipe.predict(X_test)
        
        st.subheader("Mean Square Error, MSE")
        st.markdown("MSE is the most common loss function. It's defined as Mean or average of the square of the difference between actual and estimated values. MSE is used to check how close predictions are to actual values and hence it ensures the trained model to have no outlier predictions with significant errors. Its equation is as given below. For further readings, [please click here](https://www.mygreatlearning.com/blog/mean-square-error-explained/).")
        st.markdown(r"""$MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-y_i^{-})²$""")
        st.write("For the model applied, the value obtained is:")
        rnd_MSE = mean_squared_error(y_test, y_pred)
        st.write("Mean Squared Error, MSE:", rnd_MSE)
        #Calculating the Root Mean Squared Error
        
        st.subheader("Root Mean Square Errot, RMSE")
        st.markdown("[RMSE](https://www.sciencedirect.com/topics/engineering/root-mean-squared-error) is the standatd deviation of the residual(prediction errors. It measures how far the residuals are from the regresssion line data. In short, it shows how concentrated the data is around the line of best fit. When standarized observations($O_i$) are predictions($S_i$) are used as RMSE inputs, there is a direct relationship with the correlationn coefficient. For example, if the correlation is 1, the RMSE will be 0, because all of the points lie on the regression line and hence there is no error. ")
        st.markdown(r"""$RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (S_i-O_i)^{2}}$""")
        rnd_RMSE = np.sqrt(rnd_MSE)
        st.write("Root Mean Squared Error, RMSE:", rnd_RMSE)


        st.subheader("Accuracy, Precision and Recall")
        accuracy = pipe.score(X_test, y_test)
        st.write("Accuracy:", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, average='weighted').round(2))
        st.write("Recall: ", recall_score(y_test, y_pred,average='weighted').round(2))
        #classification Report
        #st.write(metrics.classification_report (y_test, np.argmax(y_pred, axis = 1))) 
         
#------------------------------------------------------------
from sklearn.inspection import permutation_importance  
from sklearn import svm

with pca_data:
    st.subheader("Principal Component Ananlysis(PCA)")

    st.markdown("[Lerma, 2019](https://sites.math.northwestern.edu/~mlerma/papers/princcomp2d.pdf) defines Principal component analysis (PCA) as a mathematical procedure intended to replace a number of correlated variables with a new set of variables that are linearly uncorrelated. It's is an [unsupervised machine learning technique](https://machinelearningmastery.com/principal-component-analysis-for-visualization/). It can be used as a data preparation technique and vizualizing data. As  [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) allows to summarize the information content in the data tables by means of a smaller set of 'summary indices' that can be easily visualized and analyzed. It does and simplifies mathematical concepts such as standardization, covariance, eigenvectors and eigenvalues without focusing on how to compute them. One of its principal component is reduction of dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains the most information in large set. Hence it is an important approach to simplify the understanding about the dataset for machine learning. For more, [please click this link](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)")
    st.markdown("Steps to compute PCA:")
    st.write("1. Standardization")
    st.write("2. Covariance matrix computation")
    st.write("3. Compute the eigenvectors and eigenvalues of the covariance matrix to identify the principal components")
    st.write("4. Feature vector")
    st.write("5. Recast the data along the principal components axes")
    st.markdown("For more details you can read an article written by Zakaria Jaadi(2001), [A Step-by-Step Explanation of Principal Component Analysis (PCA)(https://builtin.com/data-science/step-step-explanation-principal-component-analysis)]. He has elaborated each steps in detail. ")

    st.header("Extraction of Features of Contribution")
    st.subheader("Best Features Extraction")
    if classifier_name == "Random Forest Regressor":

        
        if dataset_name == "Dolphins":
            #categorical features
            categorical_features = X.select_dtypes(include=['object']).columns
            st.write(categorical_features)

            #onehot encoder

            onehot_features = list(pipe.named_steps['preprocessor'].named_transformers_['data_cat'].named_steps['onehot'].get_feature_names_out(input_features=categorical_features))
            st.write(onehot_features)
            #numerical features
            numerical_features = X.select_dtypes(exclude=['object']).columns
            st.write(numerical_features)

            #all features of interest

            all_features = list(onehot_features) + list(numerical_features)
            st.write(all_features)
                
            #importance = pipe.named_steps['model'].feature_importances
            importance = permutation_importance(pipe, X_test, y_test).importances_mean
            st.write(importance)    

            zipped = zip(all_features, importance)
            df = pd.DataFrame(zipped, columns=["all_features", "importance"])
            df = df.sort_values("importance", ascending=False)
            st.write(df)

            # Sort the features by the absolute value of their coefficient

            #Graphic representation
            fig = plt.figure()
            sns.barplot(y=df['all_features'], x=df['importance'], data=df)
            st.pyplot(fig)
        else:
            #categorical features
            #onehot = OneHotEncoder(handle_unknown='ignore', sparse = False)
            #onehot.fit(X)

            
            numerical_features = X.select_dtypes(exclude=['object']).columns
            st.write(numerical_features)

            #all features of interest

            all_features = list(numerical_features)
            st.write(all_features)
                
            #importance = pipe.named_steps['model'].feature_importances
            importance = permutation_importance(pipe, X_test, y_test).importances_mean
            st.write(importance)    

            zipped = zip(all_features, importance)
            df = pd.DataFrame(zipped, columns=["all_features", "importance"])
            df = df.sort_values("importance", ascending=False)
            st.write(df)

            # Sort the features by the absolute value of their coefficient

            #Graphic representation
            fig = plt.figure()
            sns.barplot(y=df['all_features'], x=df['importance'], data=df)
            st.pyplot(fig)

    else:
            if dataset_name == "Dolphins":
                #categorical features
                categorical_features = X.select_dtypes(include=['object']).columns
                st.write(categorical_features)

                #onehot encoder

                onehot_features = list(pipe.named_steps['preprocessor'].named_transformers_['data_cat'].named_steps['onehot'].get_feature_names_out(input_features=categorical_features))
                st.write(onehot_features)
                #numerical features
                numerical_features = X.select_dtypes(exclude=['object']).columns
                st.write(numerical_features)

                #all features of interest

                all_features = list(onehot_features) + list(numerical_features)
                st.write(all_features)
                    
                importance = permutation_importance(pipe, X_test, y_test, scoring='accuracy').importances_mean
                st.write(importance)    

                zipped = zip(all_features, importance)
                df = pd.DataFrame(zipped, columns=["all_features", "importance"])
                df = df.sort_values("importance", ascending=False)

                # Sort the features by the absolute value of their coefficient

                #Graphic representation
                fig = plt.figure()
                sns.barplot(y=df["all_features"], x=df["importance"], data=df)
                st.pyplot(fig)   
            else:
                #categorical features
                    
                numerical_features = X.select_dtypes(exclude=['object']).columns
                st.write(numerical_features)

                #all features of interest

                all_features = list(numerical_features)
                st.write(all_features)
                    
                importance = permutation_importance(pipe, X_test, y_test, scoring='accuracy').importances_mean
                st.write(importance) 

                zipped = zip(all_features, importance)
                df = pd.DataFrame(zipped, columns=["all_features", "importance"])
                df = df.sort_values("importance", ascending=False)

                # Sort the features by the absolute value of their coefficient

                #Graphic representation
                fig = plt.figure()
                sns.barplot(y=df["all_features"], x=df["importance"], data=df)
                st.pyplot(fig)

#------------------------------
from sklearn.decomposition import PCA

#if classifier_name == "Random Forest Classifier":
df_reduced = PCA(n_components=2).fit_transform(X.dropna().select_dtypes(exclude=['object']))
st.write(df_reduced)

pca = PCA(n_components=2).fit(X.dropna().select_dtypes(exclude=['object']))
st.write(pca.explained_variance_ratio_)
#plotting scatter diagram

st.subheader("Scatter plot of high dimensional data")
st.write("The data input is tranformed X by PCA into df_reduced. Only the first two columns with most features importance are used for plotting 2D scatter")
if dataset_name == "Dolphins":
    fig = plt.figure(figsize=(14, 10))
    sns.scatterplot(x=df_reduced[:, 0], y=df_reduced[:, 1], hue = data.dropna()['variety'], s=30)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    st.pyplot(fig)
elif dataset_name == "Wine Quality":
    fig = plt.figure(figsize=(14, 10))
    sns.scatterplot(x=df_reduced[:, 0], y=df_reduced[:, 1], hue = data.dropna()['type'], s=30)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    st.pyplot(fig)
elif dataset_name == "Iris":
    fig = plt.figure(figsize=(14, 10))
    sns.scatterplot(x=df_reduced[:, 0], y=df_reduced[:, 1], hue = data.dropna()['Species'], s=30)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    st.pyplot(fig)
elif dataset_name == "Breast cancer":
    fig = plt.figure(figsize=(14, 10))
    sns.scatterplot(x=df_reduced[:, 0], y=df_reduced[:, 1], hue = data.dropna()['diagnosis'], s=30)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    st.pyplot(fig)
elif dataset_name == "Spam classifier":
    fig = plt.figure(figsize=(14, 10))
    sns.scatterplot(x=df_reduced[:, 0], y=df_reduced[:, 1], hue = df_perc.dropna()['Label'], s=30)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    st.pyplot(fig)

    







