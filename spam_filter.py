import pandas as pd
import numpy as np
import seaborn as sns
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

data=pd.read_csv('spam.csv',encoding='latin-1')

data.dropna(how="any", inplace=True, axis=1)
data.columns=['Category','Message']
data['Category']=data['Category'].map({'ham':0,'spam':1})
data.head()
Original=data.copy()

def clean_msg(message):
    ''' Inputs the messages in the data
    1. Removes stopwords
    2. Removes punctuations
    Returns filtered message as a string'''
    stop_words=set(stopwords.words('english'))
    msg= message
    words=word_tokenize(msg.lower())
    filtered_msg=[]
    for word in words:
        if(word not in stop_words and word.isalnum()):
            filtered_msg.append(word)
    return(' '.join(filtered_msg))
    

data['filt_Message']=data.Message.apply(clean_msg)

def totalwords(message):
    return(len(message.split()))
data['Totalwords']=data['Message'].apply(totalwords)


X=data.drop(['Message','Category'],axis=1)
Y=data['Category']


from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer(stop_words='english',ngram_range=(1, 2))
features=vectorizer.fit_transform(data.filt_Message)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, Y,test_size=0.3,random_state=1)


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)

def spam_filter(Example):
    Example_clean=[clean_msg(Example)]
    x=vectorizer.transform(Example_clean)
    if nb.predict(x)==1:
        print('It is a spam')
    else:
        print('Its not a spam')
