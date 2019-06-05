import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import pickle
import nltk
#nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """
    this function loads data from database file
    Parameters:
        database_filepath (string type): file path of the input database
    Returns:
        X (dataframe): feature data
        Y (dataframe): label data
        category_names (list): labels
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("Message_new", con=engine)
    X = df["message"]
    y = df.drop(['message', 'genre', 'id', 'original'], axis = 1)
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    This function lemmatizes and tokenizes text in the messages
    
    Parameters:
    text (string): the input text
    
    Returns:
    clean_tokens (list): a list of lemmatized and tokenized strings from a message text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    This function build a machine learning model with grid search and NPL pipeline
    
    Parameters: none
    
    Returns: a machine learning estimator with NPL pipeline processes and specified parameters.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('best', TruncatedSVD()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    parameters = {
        'tfidf__use_idf': (True, False), 
     'clf__estimator__learning_rate': [1,2]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates machine learning models
    
    Parameters:
    model (estimator): the machine learning estimator after training
    X_test (dataframe): the test dataset for prediction
    Y_test (dataframe): the real dataset
    category_names (lsit): the input for target_names in classification report
    
    Returns: none, but there is a report that shows the performance of the machine learning to print
    """
    
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, Y_test, target_names = category_names))



def save_model(model, model_filepath):
    """
    This function saves the trained machine learning model to a targeted file path
    
    Parameters:
    model (estimator): the machine learning estimator after training
    model_filepath (string): the desired file path for meachine learning model
    
    Returns: none, but the trained machine learning model is saved as a pickle file
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
