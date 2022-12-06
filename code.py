from msilib.schema import Directory
import streamlit as st
from xml.dom.minidom import Document
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from typing import final
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.corpus.reader import NOUN
import scipy.sparse
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from regex import B
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from scipy.sparse import hstack
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import metrics
import re
# import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import math
import pandas as pd
from pathlib import Path  
import os
import json
import csv
from itertools import chain
from turtle import pos, position
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('wordnet_ic')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
corpus=[]
noun_list=[]
lexical_chain=[]
count=0
i=0
ps = PorterStemmer()
docList=0
doc_csv=[]
course=[]
bigrams=[]
new_bigram_list=[]
stop_words = set(stopwords.words('english'))
field_name=["Count","Documents","Pre-Processed Text"]

while count != 1051:   #Total 1051 Documents, so looping through them to process them
    if count==230:      #230 course, documents, so directory change after 230 documents
        flag=1
        directory='D:/UNIVERSITY/SEMESTER 6/IR/A3/course-cotrain-data/fulltext/non-course'
        flag=flag+1
    if count==0:
        directory='D:/UNIVERSITY/SEMESTER 6/IR/A3/course-cotrain-data/fulltext/course'
    for filename in os.listdir(directory):   #extract the filename
            i=i+1
            count=count+1
            if(count<=230):     #keeping count for course and non-course documents
                course.append(1)
            elif(count>230):
                course.append(0)
            text_file_name=os.path.join(directory,filename)   #joining directory path with filename
            text_titles = [Path(text).stem for text in text_file_name]
            html_page = open(os.path.join(directory, filename), "r")  
            soup = BeautifulSoup(html_page, "html.parser")  #reading html docs
            html_text = soup.get_text()
            doc_csv.append(html_text)
            docList = html_text
            docList = docList.replace("\n", " ")    ##Pre-Processing, removing lines,tabs,etc
            docList = docList.replace('-', " ")
            docList = docList.replace("/", " ")
            docList = docList.replace("/t"," ")
            tokens = nltk.word_tokenize(docList)   #Tokenising docs
            tokens = [tokens.lower() for tokens in tokens if tokens.isalnum()]  #keeping only letters and numbers, remvoing all special characters
            filtered_sentence = [w for w in tokens if not w.lower() in stop_words] #Stop words removed
            filtered_sentence = []
            for w in tokens:
                if w not in stop_words:
                    stem_word=ps.stem(w)   #stemming
                    filtered_sentence.append(stem_word) 
            bigrams.append(list(nltk.bigrams(filtered_sentence))) #Making Bi-grams needs in part2
            corpus.append(filtered_sentence)  #Pre-Processed text stored in corpus list, document wise
count=0            
print("This is doc:",len(doc_csv))
print("This is fil:",len(corpus))
print("Ttal Coun:",i)
with open("dataset.csv",mode="w") as csvfile:  #storing data into a csv file
    fieldnames=["Count","Document","Pre-Process","Bigrams","Course"]
    writer=csv.DictWriter(csvfile,fieldnames=fieldnames)
    writer.writeheader()
    while(count!=i):
        writer.writerow({"Count":count+1,"Document":doc_csv[count],"Pre-Process":corpus[count],"Bigrams":bigrams[count],"Course":course[count]})
        count=count+1
csvfile.close()
df = pd.read_csv('dataset.csv',encoding = 'ISO-8859-1')
#Model for part I
vectorizer = TfidfVectorizer(min_df=1,max_df=100,tokenizer=lambda doc: doc, lowercase=False) #TFIDF Library intialised
X1 = vectorizer.fit_transform(corpus)  #does all the tfidf work
print(X1.toarray())
#Model
X_train, X_test, Y_train, Y_test = train_test_split(X1.toarray(), df['Course'], test_size=0.3, random_state=6)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train).predict(X_test)
acc1=metrics.accuracy_score(Y_test,y_pred)
rec1=metrics.recall_score(Y_test, y_pred,average='weighted')
pre1=metrics.precision_score(Y_test, y_pred,average='weighted')
f1=metrics.f1_score(Y_test, y_pred,average='weighted')
print("Part I Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print("Part I Recall:",metrics.recall_score(Y_test, y_pred,average='weighted'))
print("Part I Precision:",metrics.precision_score(Y_test, y_pred,average='weighted'))
print("Part I F1-Score:",metrics.f1_score(Y_test, y_pred,average='weighted'))

#Part 2 Starts here
nouns = [word for (word, pos) in nltk.pos_tag(df['Pre-Process']) if(pos[:2] == 'NN')] #Taking all the nouns from the pre-processed text
for i in range(len(nouns)):
    res = nouns[i].strip('][').split(', ')   #converting a string list to list
    noun_list.append(res)
new_noun_list = list(chain.from_iterable(noun_list)) #converting 2d list to 1d list
noun_count = nltk.FreqDist(new_noun_list) #frequency of each noun in whole corpus
mostcommon = noun_count.most_common(50) #picking top50 most common nouns
f = open("noun_count.txt", "w")
f.write(str(noun_count.items()))
f.close()
print(len(mostcommon))
f = open("most_common.txt", "w")
f.write(str(mostcommon))
f.close()
count1=-1
temp_list=[]
for bi in bigrams:   #Finding bigrams which appear in noun list, to pass them to our model, storing them in a new list
    temp_list=[]
    for bi1 in bi:
        for mc in mostcommon:
            x=mc[0].replace("'",'')
            if(x==bi1[0] or x==bi1[1]):
                temp_list.append(bi1)
    new_bigram_list.append(temp_list)
print(len(new_bigram_list))
f = open("new_bigram_list.txt", "w")
f.write(str(new_bigram_list))
f.close()
label_encoder = preprocessing.LabelEncoder()   #Encoding, since model only undestands numbers
X2=label_encoder.fit_transform([str(t) for t in new_bigram_list]) 
X2=X2.reshape(-1,1)
print(X2)
X_train, X_test, Y_train, Y_test = train_test_split(X2, df['Course'], test_size=0.3, random_state=1)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train).predict(X_test)
acc2=metrics.accuracy_score(Y_test,y_pred)
rec2=metrics.recall_score(Y_test, y_pred,average='weighted')
pre2=metrics.precision_score(Y_test, y_pred,average='weighted')
f2=metrics.f1_score(Y_test, y_pred,average='weighted')
print("Part II Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print("Part II Recall:",metrics.recall_score(Y_test, y_pred,average='weighted'))
print("Part II Precision:",metrics.precision_score(Y_test, y_pred,average='weighted'))
print("Part II F1-Score:",metrics.f1_score(Y_test, y_pred,average='weighted'))

# part3 starts here

new_lexical_noun_list=[]
final_list=[]
lexical_nouns = [word for (word, pos) in nltk.pos_tag(df['Pre-Process']) if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')] #Creating nouns for lexical chain
for i in range(len(lexical_nouns)):
    res = lexical_nouns[i].strip('][').split(', ') #String to list
    new_lexical_noun_list.append(res)
final_list = list(chain.from_iterable(new_lexical_noun_list)) #2d to 1d list
for nouns in final_list:
    temp_list=[]
    x=nouns.replace("'",'')         #taking out hyponyms and hypernyms needed to form chain
    temp=wn.synsets(x, pos=wn.NOUN)
    for i in temp:
        hyp1=i.hyponyms()
        temp_list.append(hyp1)
    for i in temp:
        hyp=i.hypernyms()
        temp_list.append(hyp)
    lexical_chain.append(temp_list)  #making chain
    
path_similar=[]
lch_similar=[]
wup_similar=[]
new_list=[]
count=0
for lex in lexical_chain:   #finding words occuring in both list, and storing in a new list
    for noun in lexical_nouns:
        if(lex==noun):
            new_list.append(noun)


for i in range(len(new_list)-1):  #Path wise similarity for all co-occurences in new list
    x=new_list[i].replace("'",'')
    y=new_list[i+1].replace("'",'')
    var1=wn.synsets(x, pos=wn.NOUN)
    var2=wn.synsets(y, pos=wn.NOUN)
    for i in var1:
        for j in var2:
            sim1=i.path_similarity(j)
            sim2=j.path_similarity(i)
            path_similar.append([sim1,sim2])

for i in range(len(new_list)-1): #wup wise similarity for all co-occurences in new list
    x=new_list[i].replace("'",'')
    y=new_list[i+1].replace("'",'')
    var1=wn.synsets(x, pos=wn.NOUN)
    var2=wn.synsets(y, pos=wn.NOUN)
    for i in var1:
        for j in var2:
            sim1=i.wup_similarity(j)
            sim2=j.wup_similarity(i)
            sim3=wn.wup_similarity(i,j)  #Combine Similarity
            if(sim1>=0.5):
                wup_similar.append(sim1) #list containing wup similarities between words
            elif(sim2>=0.5):
                wup_similar.append(sim2)

for i in range(len(new_list)-1): #lch similarity for all co-occurences in new list
    x=new_list[i].replace("'",'')
    y=new_list[i+1].replace("'",'')
    var1=wn.synsets(x, pos=wn.NOUN)
    var2=wn.synsets(y, pos=wn.NOUN)
    for i in var1:
        for j in var2:
            sim1=i.lch_similarity(j)
            sim2=j.lch_similarity(i)
            sim3=wn.lch_similarity(i,j) #Combine Similarity
            lch_similar.append([sim1,sim2,sim3]) #list containing lch similarities between words
label_encoder = preprocessing.LabelEncoder()
X3=label_encoder.fit_transform([str(t) for t in wup_similar])
X3=X1.toarray()
X_train, X_test, Y_train, Y_test = train_test_split(X3, df['Course'], test_size=0.3, random_state=3)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train).predict(X_test)
acc3=metrics.accuracy_score(Y_test,y_pred)
rec3=metrics.recall_score(Y_test, y_pred,average='weighted')
pre3=metrics.precision_score(Y_test, y_pred,average='weighted')
f3=metrics.f1_score(Y_test, y_pred,average='weighted')
print("\nPart III Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print("Part III Recall:",metrics.recall_score(Y_test, y_pred,average='weighted'))
print("Part III Precision:",metrics.precision_score(Y_test, y_pred,average='weighted'))
print("Part III F1-Score:",metrics.f1_score(Y_test, y_pred,average='weighted'))


# acc4=acc1+acc2+acc3
# result=acc4/3
# rec4=rec1+rec2+rec3
# result_rec=rec4/3
# pre4=pre1+pre2+pre3
# result_pre=rec4/3
# f4=f1+f2+f3
# result_f4=f4/3
# print("\nCombine Accuracy:",result)
# print("Combine Recall:",result_rec)
# print("Combine Precision:",result_pre)
# print("Combine F1-Score:",result_f4)

#Combine Accuracy
X4 = scipy.sparse.hstack([X1,X2,X3])
X_train, X_test, Y_train, Y_test = train_test_split(X4.toarray(), df['Course'], test_size=5, random_state=3)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train).predict(X_test)
com_acc=metrics.accuracy_score(Y_test,y_pred)
com_f1=metrics.f1_score(Y_test, y_pred,average='weighted')
X_train, X_test, Y_train, Y_test = train_test_split(X4.toarray(), df['Course'], test_size=0.3, random_state=3)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train).predict(X_test)
com_rec=metrics.precision_score(Y_test, y_pred,average='weighted')
com_precsion=metrics.precision_score(Y_test, y_pred,average='weighted')



##UI CODE
st.title("Part I")
Accuracy,Recall,Precsion,Fscore=st.columns(4)
Accuracy.subheader("Accuracy")
Accuracy.text(acc1)
Recall.subheader("Recall")
Recall.text(rec1)
Precsion.subheader("Precision")
Precsion.text(pre1)
Fscore.subheader("F-Score")
Fscore.text(f1)

st.title("Part II")
Accuracy,Recall,Precsion,Fscore=st.columns(4)
Accuracy.subheader("Accuracy")
Accuracy.text(acc2)
Recall.subheader("Recall")
Recall.text(rec2)
Precsion.subheader("Precision")
Precsion.text(pre2)
Fscore.subheader("F-Score")
Fscore.text(f2)

st.title("Part III")
Accuracy,Recall,Precsion,Fscore=st.columns(4)
Accuracy.subheader("Accuracy")
Accuracy.text(acc3)
Recall.subheader("Recall")
Recall.text(rec3)
Precsion.subheader("Precision")
Precsion.text(pre3)
Fscore.subheader("F-Score")
Fscore.text(f3)

st.title("Combined Scores")
Accuracy,Recall,Precsion,Fscore=st.columns(4)
Accuracy.subheader("Accuracy")
Accuracy.text(com_acc)
Recall.subheader("Recall")
Recall.text(com_rec)
Precsion.subheader("Precision")
Precsion.text(com_precsion)
Fscore.subheader("F-Score")
Fscore.text(com_f1)
            




