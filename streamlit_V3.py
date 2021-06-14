# -*- coding: utf-8 -*-

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import NMF
# import pandas as pd
import streamlit as st
import numpy as np
import nltk
import spacy
import gensim
import joblib

import seaborn as sns
import matplotlib.pyplot as plt



# Mettre en minuscul
def lowed_text(text_to_clean):
    lower_text = []
    for doc in text_to_clean:
        lower_text.append(doc.lower())

    return lower_text


# tokenisation
def tokeniser(lower_text):
    tokenized_text = []
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    for doc in lower_text:
        tokenized_text.append(tokenizer.tokenize(doc))

    return tokenized_text


# lemmatisation
def lemmatation(text_tekenized, nlp):
    lemma_text = []
    for elm1 in text_tekenized:
        r1 = nlp(" ".join(elm1))
        lemma_text.append([token.lemma_ for token in r1])

    return(lemma_text)


# Supression stop_word
def stop_word(text_lemated, stop_words):
    def valid_word(word):
        return word not in stop_words and not word.isdecimal() and len(word) > 1

    # Supression stop_word
    filtered_docs = []
    for doc_tokens in text_lemated:
        filtered_docs.append(
            [token for token in doc_tokens if valid_word(token)])

    return filtered_docs


def dummy_fun(doc):
    return doc


def term_tf_idf(filtered_docs, tf_idf):
    terms = tf_idf.transform(filtered_docs)
    return terms


# Model des topics avec NMF
def topic_model(nmf, tf_idf_fited, terms, top_words=10):
    # trouver le topic le plus probable pour chaque sentence
    # afficher les N mots les plus probables associés à chacun des topics
    res = nmf.transform(tf_idf_fited)
    most_likely_topic_id = np.argmax(res)
    best_inds = np.argsort(nmf.components_[most_likely_topic_id])[
        ::-1][:top_words]
    sns.barplot(nmf.components_[most_likely_topic_id][best_inds], [
        terms[ind] for ind in best_inds])
    plt.title('Classement des mots du topic')
    st.pyplot()
    return [terms[ind] for ind in best_inds]


def main():
    # df_bad_review = pd.read_csv('df_bad_review.csv')
    nltk.download('stopwords')
    stop_words = list(nltk.corpus.stopwords.words('english'))
    stop_words += ['would', 'order', 'go', 'sit',
                   'I', 'st', 'think', 'yes', 'yet', 'yelp']
    
    nlp = spacy.load('en')
    tf_idf = joblib.load('./model_tf_idf.joblib')
    terms = tf_idf.get_feature_names()
    nmf = joblib.load('./model_nmf.joblib')

    st.title('Analyse de sentiments négatifs')
    user_input = st.text_area("", "Saisir le texte à analyser (en anglais)")
    predict = st.button('Prédire')

    if predict:
        text_lowed = lowed_text([user_input])
        text_tekenized = tokeniser(text_lowed)
        text_lemated = lemmatation(text_tekenized, nlp)
        filtered_docs = stop_word(text_lemated, stop_words)
        tf_idf_fited = term_tf_idf(filtered_docs, tf_idf)

        st.write('Voici la description du topic le plus probable pour cette entrée :')
        model_topic = topic_model(nmf, tf_idf_fited, terms)


if __name__ == '__main__':
    main()
