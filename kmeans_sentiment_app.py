# Import Library
import pandas as pd
import numpy as np
import pickle as pkl
import nltk
import streamlit as st
import streamlit_authenticator as stauth
from sklearn.mixture import GaussianMixture

# Load Model
filename = 'finalized_model_kmeans_sentiment.sav'
model = pkl.load(open(filename, 'rb'))

# Create a GMM Model
clf = GaussianMixture(n_components=3, covariance_type='spherical',
                      means_init = model.cluster_centers_ ,n_init= 1 ,max_iter=1)

# Create a BoW Model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=5)

# Create Simple Login Authenticator
names = ['Miranti Alysha', 'Admin DB']
usernames = ['miwa', 'admin']
passwords = ['tytrack10', '12345']

hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(names,usernames,hashed_passwords,
                                    'some_cookie_name','some_signature_key',cookie_expiry_days=30)

name, authentication_status, username = authenticator.login('Login','main')

if authentication_status:
    # Create a Web App
    st.header('Sentiment Analysis Clustering with K-Means Algorithm')

    st.write('##### Author : Miwa')

    st.text('''
    This web application is an application that will perform sentiment analysis on a
    text and then group it into text with positive, negative, or neutral sentiments.''')

    st.write('This application is made using **Python** and **Streamlit**')

    st.write(
        "Google Colab Notebook [Click Me!](https://colab.research.google.com/drive/1omy7beT8UbkvT3FTnS8NLKdobsvPVMQN?usp=sharing)")

    text = st.text_input('Input your Text', 'Thank you K-Pop karena sudah mengubah duniaku menjadi lebih baik')

    if text:
        if len(text.split()) < 5:
            st.write("Error! Silakan masukkan kalimat dengan jumlah kata lebih dari 5!")
        else:
            lst = []
            lst.append(text)
            dt = pd.DataFrame()
            dt['Text'] = lst
            cv_pred = TfidfVectorizer(max_features=5)
            word_pred = cv_pred.fit_transform(dt['Text']).toarray()
            res = model.predict(word_pred)
            lst_res = []
            if res == 0:
                lst_res.append('Negative')
            elif res == 1:
                lst_res.append('Neutral')
            elif res == 2:
                lst_res.append('Positive')
            else:
                lst_res.append("Text cann't be classified")

            dt['Sentiment'] = lst_res
            st.write(dt)
    authenticator.logout("Log Out")
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')

# # Create a Web App
# st.header('Sentiment Analysis Clustering with K-Means Algorithm')
#
# st.write('##### Author : Miwa')
#
# st.text('''
# This web application is an application that will perform sentiment analysis on a
# text and then group it into text with positive, negative, or neutral sentiments.''')
#
# st.write('This application is made using **Python** and **Streamlit**')
#
# st.write("Google Colab Notebook [Click Me!](https://colab.research.google.com/drive/1omy7beT8UbkvT3FTnS8NLKdobsvPVMQN?usp=sharing)")
#
# text = st.text_input('Input your Text', 'Thank you K-Pop karena sudah mengubah duniaku menjadi lebih baik')
#
# if text:
#     if len(text.split()) < 5:
#         st.write("Error! Silakan masukkan kalimat dengan jumlah kata lebih dari 5!")
#     else:
#         lst = []
#         lst.append(text)
#         dt = pd.DataFrame()
#         dt['Text'] = lst
#         cv_pred = TfidfVectorizer(max_features=5)
#         word_pred = cv_pred.fit_transform(dt['Text']).toarray()
#         res = model.predict(word_pred)
#         lst_res = []
#         if res == 0:
#             lst_res.append('Negative')
#         elif res == 1:
#             lst_res.append('Neutral')
#         elif res == 2:
#             lst_res.append('Positive')
#         else:
#             lst_res.append("Text cann't be classified")
#
#         dt['Sentiment'] = lst_res
#         st.write(dt)