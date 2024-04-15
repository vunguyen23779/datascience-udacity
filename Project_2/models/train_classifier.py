# import libraries
import sys
import pandas as pd
import numpy as np

import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle


nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """
    This function will load data from Database
    
    :params: database_filepath
    :ouputs: X - model features, Y - model target
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_messages_tb", con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X,Y


def tokenize(text):
    """
    Given text string will be tokenized using this function
    
    :params: text
    :outputs: clean_tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens=[]
    for token in tokens:
        buffer = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(buffer)
        
    return clean_tokens


def build_model():
    """
    This function uses GridSearchCV to generate a classifier and fine-tune the model.
    :params: None
    :outputs: cv
    """    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
        
    parameters = {
        'clf__estimator__n_estimators' : [50, 100]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Evaluates model.
    
    :params: model - Classifier, X_test - test dataset, Y_test - labels
    :outputs:  Report
    """
    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    """ Exports a pickle file as the finished model.."""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ Constructs, Trains, Evaluates, and Save the model.."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
