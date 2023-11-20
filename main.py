# -*- coding: utf-8 -*-
"""
@author: ls
"""

import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#load data
path = 'dataset'
filelist = os.listdir(path)
products = pd.DataFrame()
reviews = pd.DataFrame()
for item in filelist:
    path_now = path+'/'+item
    df_now =  pd.read_csv(path_now, delimiter='\t')
    if len(item) == 19:#product file
        df_now.columns = ["id", "category", "product_title"]#column name
        products = pd.concat([products, df_now])    
    else:#review file
        if item == 'reviews-2.tsv':
            df_now.columns = ["rating", "id", "review_text"]#column name
            df_now = df_now[["id", "rating", "review_text"]]#change order
        else:
            df_now.columns=["id", "rating", "review_text"]#column name
        reviews = pd.concat([reviews, df_now])

#merge product data and review data through id
data = pd.merge(products, reviews, on = 'id')

#combine text features and rating
data['text'] = data['rating'].astype(str)+' ' + data['product_title'] + ' ' + data['review_text']
data.replace(['Kitchen','Ktchen','Jewelry'], [0,0,1], inplace = True)#label encode
x_data = data['text']
y_data = data['category']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=24)#data split

#define pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression())
])

#train model
pipeline.fit(x_train, y_train)

#evaluate model
y_pred = pipeline.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)#metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)
print("f1 score:", f1)
print("confusion matrix:", cm_norm)

