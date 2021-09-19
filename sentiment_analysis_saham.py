# -*- coding: utf-8 -*-
"""
@Title: SENTIMENT ANALYSIS ABOUT COMPANY AT MARKET STOCK ON INDONESIA PORTAL NEWS USING K-NN METHOD
@author: Agus Tri Haryono
@Description: Implemented text analysis using machine learning models to classify company at market stock as positive or negative.
"""

#Importing Essentials
import re
import nltk
import pathlib
import pandas as pd
#from sklearn.svm import LinearSVC
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.linear_model import LogisticRegression

Pathloc = str(pathlib.Path().absolute())
path = Pathloc + '\scrape_news.tsv'
data = pd.read_table(path,header=None,skiprows=1,names=['Sentiment','Title'])

def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    # to lowecase
    text = text.lower()
    return text

def clean_list(list):
    list_new = []
    for r in list:
        list_new.append(clean_text(r))
    return list_new

# pre processing case folding, remove punctuation, tokenization, stopwords
text = clean_list(data.Title)
words = [nltk.word_tokenize(word) for word in text]

stop_words = []
f = open(Pathloc + '\stopwords_indonesia.txt', 'r')
for l in f.readlines():
    stop_words.append(l.replace('\n', ''))
stop_words.append('sesi')

for i in range(len(words)):
    words[i] = [w for w in words[i] if w not in stop_words and len(w)>1]
    
# word cloud and graph analytic
import itertools
from collections import Counter
from matplotlib import pyplot as plt
from wordcloud import WordCloud

word_list = list(itertools.chain.from_iterable(words))
word_count = Counter(word_list)
words_unique = pd.DataFrame.from_dict(word_count, orient='index').reset_index()
words_unique = words_unique.rename(columns={'index':'words', 0:'counts'})

fig, ax = plt.subplots(figsize =(16, 9))
ax.barh(words_unique[words_unique['counts']>2]['words'], words_unique[words_unique['counts']>2]['counts'])
 
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
 
# Remove x, y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
 
# Add padding between axes and labels
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)
 
# Add x, y gridlines
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
 
# Show top values
ax.invert_yaxis()
 
# Add annotation to bars
for i in ax.patches:
    plt.text(i.get_width()+0.2, i.get_y()+0.5,
             str(round((i.get_width()), 2)),
             fontsize = 10, fontweight ='bold',
             color ='grey')
 
# Add Plot Title
ax.set_title('Graph Word Cloud',
             loc ='center', )
 
# Show Plot
plt.show()

print(Counter(word_list))
wordcloud = WordCloud(
    background_color='white',
    width=1600,
    height=800,
    random_state=21,
    colormap='jet',
    max_words=50,
    max_font_size=200).generate_from_frequencies(Counter(word_list))
plt.figure(figsize=(16, 9))
plt.axis('off')
plt.imshow(wordcloud, interpolation="bilinear")

# feature extraction: word2vec
from gensim.models import Word2Vec
import numpy as np
word2vec = Word2Vec(words, min_count=1, size=100)
vocab = word2vec.wv.vocab
print(vocab)
print(word2vec['naik'])
print(word2vec.similar_by_word('merah'))

wv_matrix = []
for word in words_unique['words']:  
    wv_matrix.append([word] + word2vec[word].tolist())
    
head = ['word']
for i in range(1,101):
    head.append('v'+str(i))
    
wv_matrix = pd.DataFrame(wv_matrix, columns = head)

train_dataset = []
for i in range(len(words)):
    train_dataset.append(np.mean(word2vec[words[i]],axis=0))

# KNN Method
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Split to train and test data
X_train, X_test, y_train, y_test = train_test_split(train_dataset,data.Sentiment,random_state=1, test_size= 0.2, stratify=data.Sentiment)

#from sklearn.feature_extraction.text import CountVectorizer
##Using CountVectorizer to convert text into tokens/features
#vect = CountVectorizer(ngram_range = (1,1), max_df = .90, min_df = 4)
#X_train, X_test, y_train, y_test = train_test_split(data.Title.str.lower(),data.Sentiment,random_state=1, test_size= 0.2, stratify=data.Sentiment)
##Using training data to transform text into counts of features for each message
#vect.fit(X_train)
#X_train_dtm = vect.transform(X_train) 
#X_test_dtm = vect.transform(X_test)

#Initializing lists to be used in plotting later.
acc_list = []
rec_list = []
prec_list = []
f1_list = []

#KNN Model
for k in range(1,10):
    KNN = KNeighborsClassifier(n_neighbors = k)
    KNN.fit(X_train, y_train)
    y_pred = KNN.predict(X_test)
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

acc_wv = acc_list

plt.figure(figsize=(16, 9))
plt.plot(range(1,10),acc_list)
plt.title('Accuracy Graph')
plt.xlabel('Number of n')
plt.ylabel('Accuracy Score')
plt.show()