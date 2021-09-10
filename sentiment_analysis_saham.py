# -*- coding: utf-8 -*-
"""
@Title: SENTIMENT ANALYSIS ABOUT COMPANY AT MARKET STOCK ON INDONESIA PORTAL NEWS USING K-NN METHOD
@author: Agus Tri Haryono
@Description: Implemented text analysis using machine learning models to classify company at market stock as positive or negative.
"""

#Importing Essentials
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

path = 'D:\Documents\KULIAH S2\RPL A Tugas Akhir\scrape_news.tsv'
data = pd.read_table(path,header=None,skiprows=1,names=['Sentiment','Title'])
X = data.Title.str.lower()
y = data.Sentiment

#Using CountVectorizer to convert text into tokens/features
vect = CountVectorizer(ngram_range = (1,1), max_df = .90, min_df = 4)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size= 0.2, stratify=data.Sentiment)
#Using training data to transform text into counts of features for each message
vect.fit(X_train)
X_train_dtm = vect.transform(X_train) 
X_test_dtm = vect.transform(X_test)

#Initializing lists to be used in plotting later.
acc_list = []
rec_list = []
prec_list = []
f1_list = []

#KNN Model
for k in [1,3,5,7,10]:
    KNN = KNeighborsClassifier(n_neighbors = k)
    KNN.fit(X_train_dtm, y_train)
    y_pred = KNN.predict(X_test_dtm)
    accuracy_score = (metrics.accuracy_score(y_test,y_pred))
    accuracy_score = (round(accuracy_score,2))*100
    acc_list.append(accuracy_score)

    confusion_mat = metrics.confusion_matrix(y_test, y_pred)
    class_report = metrics.classification_report(y_test, y_pred)

    macro_precision = (metrics.precision_score(y_test, y_pred, average='macro'))
    macro_precision = (round(macro_precision,2))*100
    prec_list.append(macro_precision)

    macro_recall = (metrics.recall_score(y_test, y_pred, average='macro'))
    macro_recall = (round(macro_recall,2))*100
    rec_list.append(macro_recall)
    
    macro_f1 = (metrics.f1_score(y_test, y_pred, average='macro'))
    macro_f1 = (round(macro_f1,2))*100
    f1_list.append(macro_f1)

    print("\n\nConfusion Matrix for k = {} is:\n".format(k))
    print(confusion_mat)
    print("\nClassification Report for k = {} is:\n".format(k))
    print(class_report)
