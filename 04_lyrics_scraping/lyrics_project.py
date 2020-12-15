import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss
import argparse

parser = argparse.ArgumentParser(description='This program predicts whether the artist is the Kooks or Alicia Keys based on lyrics')

parser.add_argument("-a", "--alpha", help="alpha - smoothing hyperparameter for Naive Bayes", type= float, default=1)
parser.add_argument("-n_est", "--n_estimators", help="Random Forest: the number of trees in the forest", type= int, default=100)
parser.add_argument("-max_d", "--max_depth", help="Random Forest: maximum depth of the tree", type= int, default=5)

args = parser.parse_args()

#scraping/parsing
def lyrics(url):
    r1=requests.get(url)
    songs=BeautifulSoup(r1.text, features="lxml")
    songs = songs.find_all('td',attrs={'class':'tal qx'})
    df = pd.DataFrame() 
    for i in songs:
        lyric_links = "https://www.lyrics.com" + i.find("a").get("href")     
        r2 = requests.get(lyric_links)
        if r2.status_code==200:
            lyrics= BeautifulSoup(r2.text,features="lxml") 
            try:
                artist = lyrics.find("h4").find("a").text
                lyrics=lyrics.find('pre', attrs={'id':'lyric-body-text', "class":"lyric-body"}).text
                lyrics = re.sub("\\n"," ",lyrics)                 
                data = pd.DataFrame({"artist": [artist], "songs": [lyrics]})
                df = df.append(data)
            except AttributeError:
                continue
    return df    

kooks=lyrics('https://www.lyrics.com/artist/The-Kooks/762797')
alicia=lyrics('https://www.lyrics.com/artist/Alicia-Keys/469431')
allsongs=kooks.append(alicia)

allsongs=allsongs[allsongs['artist'].isin(['The Kooks','Alicia Keys'])]  #deleting rows with remixes
allsongs.reset_index(inplace=True)
allsongs= allsongs.drop(["index"], axis=1)

#saving to csv
allsongs.to_csv('allsongs.csv',index=False)

df=pd.read_csv('allsongs.csv',engine='python')

# preprocessing
X = list(df['songs'])
y = list(df['artist'])

#splitting into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#spacy
nlp = spacy.load('en_core_web_md')

def clean_text(corpus, model):
    """preprocess a string (tokens, stopwords, lowercase, lemma & stemming) returns the cleaned result
        params: review - a string
                model - a spacy model                
        returns: list of cleaned strings
    """    
    new_doc = []
    doc = model(corpus)
    for word in doc:
        if not word.is_stop and word.is_alpha:
            new_doc.append(word.lemma_.lower())  
            final = ", ".join(map(str,new_doc))   
    return final

list1 = []
for i in tqdm(X_train):
    try:
        d = clean_text(i, nlp)
        list1.append(d)
    except TypeError:
        continue

X_train=list1

# TfidfVectorizer
tv = TfidfVectorizer() 
tv.fit(X_train)
tv_corpus = tv.transform(X_train)
Xtrain = pd.DataFrame(tv_corpus.todense().round(2), columns=tv.get_feature_names())
test_corpus = tv.transform(X_test)
Xtest= pd.DataFrame(test_corpus.todense().round(2), columns=tv.get_feature_names())

# Train model
def evaluate_model(m,Xtrain, y_train, Xtest, y_test):
    m.fit(Xtrain, y_train)
    print("train accuracy: ", (m.score(Xtrain, y_train) *100).round(2), " %")
# Cross validation
    val = cross_val_score(m, Xtrain, y_train, cv=5, scoring='accuracy')
    print("cross validation accuracy: ", (val * 100).round(2))
    print('\n\033[1m' + "Test Data:")
    y_pred = m.predict(Xtest)
    print('\033[0m'+"test accuracy: ", (m.score(Xtest, y_test) *100).round(2), " %")
    print("precision score: ", (precision_score(y_test, y_pred, average='weighted') *100).round(2), " %")
    print("f1 score:", (f1_score(y_test, y_pred, average='weighted') *100).round(2), " %")  
    plot_confusion_matrix(m, Xtest, y_test, normalize=None)

# Random Forest
def RandForestCl(Xtrain, y_train, Xtest, y_test, maximum_depth):    
    m = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,)  
    evaluate_model(m, Xtrain, y_train, Xtest, y_test)
RandForestCl(Xtrain, y_train, Xtest, y_test,10)

# Logistic regression
def LogReg(Xtrain, y_train, Xtest, y_test):   
    m = LogisticRegression(C=1e5)
    evaluate_model(m, Xtrain, y_train, Xtest, y_test)
LogReg(Xtrain, y_train, Xtest, y_test)

# Decision Tree
def DecisionTreeCl(Xtrain, y_train, Xtest, y_test, maximum_depth):
    m = DecisionTreeClassifier(max_depth = maximum_depth)    
    evaluate_model(m, Xtrain, y_train, Xtest, y_test)
DecisionTreeCl(Xtrain, y_train, Xtest, y_test, 10)

# Naive Bayes and Pipeline
NBpipeline = Pipeline([("TV", TfidfVectorizer()),
                       ("NaiveBayes", MultinomialNB(alpha=args.alpha))])
NBpipeline.fit(X_train, y_train)
evaluate_model(NBpipeline, X_train, y_train, X_test, y_test)

# Prediction
lr= LogisticRegression(C=1e5)
lr.fit(Xtrain, y_train)
lr.predict(Xtest)

def predict(lyrics_input):
    lyrics_list = [lyrics_input]
    test_corpus = tv.transform(lyrics_list)
    lyrics_final= pd.DataFrame(test_corpus.todense().round(2), columns=tv.get_feature_names())
    prediction = lr.predict(lyrics_final)
    print(prediction)

lyrics_input = input("Please enter some lyrics")
predict(lyrics_input)

# Undersampling
rus = RandomUnderSampler(sampling_strategy={"Alicia Keys":50})
nm = NearMiss(sampling_strategy={"Alicia Keys":50})
X_rus, y_rus = rus.fit_resample(Xtrain, y_train)
X_nm, y_nm = nm.fit_resample(Xtrain, y_train)
lr.fit(X_rus, y_rus)
ypred_rus = lr.predict(Xtest)
predict(lyrics_input)

# Oversampling
ros = RandomOverSampler(sampling_strategy={'The Kooks':382})
X_ros, y_ros = ros.fit_resample(Xtrain, y_train)
np.unique(y_ros, return_counts=True)
lr.fit(X_ros, y_ros)
predict(lyrics_input)

