"""
@author: Shri Adke
@created: 12/30/2020

@purpose: Intent classification using machine learning

@use: python IntentClassifier.py

"""

# Data Loading
import json
import pandas as pd
# Data Cleaning and preprocessing
import nltk
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
# Evaluation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class IntentClassifier:
    classifiers = {}
    supported_classifiers = {
                                "SVC" : SVC(),
                                "XGB": XGBClassifier(),
                                "NaiveBayes": MultinomialNB(),
                                "MLP": MLPClassifier()
                            }
    
    def __init__(self, datapath, intent_subset="small", classifier_labels=["SVC", "XGB", "NaiveBayes", "MLP"], train=True, evaluate=True):
        with open(datapath) as json_file:
            self.data_dict = json.load(json_file)
        
        self.intent_subset = intent_subset
        self.intent_small = ["balance", "pin_change","credit_score", "rewards_balance","calories",
                        "restaurant_reviews", "shopping_list", "next_song", "traffic","uber",
                        "translate", "vaccines","alarm", "date", "taxes", 
                        "insurance", "greeting", "fun_fact","cancel", "yes"]
        
        for label in classifier_labels:
            if label in self.supported_classifiers:
                print("Using {} classifier".format(label))
                self.classifiers[label] = self.supported_classifiers[label]
            else:
                print("Invalid classifier specified")
            
        self.train = train
        self.eval = evaluate
        
    def get_dataframe(self):
        if not self.data_dict:
            print("Data not found. Please make sure data is present in given format.")
            return
        train_data = self.data_dict['train']
        val_data = self.data_dict['val']
        test_data = self.data_dict['test']
        
        train_df = pd.DataFrame(train_data, columns =['query', 'intent'])
        val_df = pd.DataFrame(val_data, columns =['query', 'intent'])
        test_df = pd.DataFrame(test_data, columns =['query', 'intent'])
        
        if self.intent_subset == "all":
            return train_df.append(val_df).append(test_df), len(train_df), len(val_df), len(test_df)
        else:
            train_df_small = train_df[train_df['intent'].isin(self.intent_small)]
            val_df_small = val_df[val_df['intent'].isin(self.intent_small)]
            test_df_small = test_df[test_df['intent'].isin(self.intent_small)]
        
            return train_df_small.append(val_df_small).append(test_df_small), len(train_df_small), len(val_df_small), len(test_df_small)
    
    def cleaning(self, sents):
        words = []
        lemmatizer = WordNetLemmatizer()
        for sent in sents:
            # Remove anything that is not alphanumeric character
            cleaned_sent = re.sub(r'[^ a-z A-Z 0-9]', " ", sent)
            # Split sentence into chunks of words i.e. tokens
            word = nltk.tokenize.word_tokenize(cleaned_sent)
            # Lemmatize the tokens i.e. convert it to its dictionary form
            words.append([lemmatizer.lemmatize(w.lower()) for w in word])

        return words
    
    def temp_tokenizer(self, text):
        return text
    
    def vectorize_data(self, cleaned_entire_sents):
        tfidf = TfidfVectorizer(max_features=8000, lowercase=False, tokenizer=self.temp_tokenizer, ngram_range=(1,3))
        X_entire = tfidf.fit_transform(cleaned_entire_sents).toarray()
        return X_entire
    
    def split_data(self, X_entire_vectorized, entire_intents, len_train, len_val, len_test):
        X_train = X_entire_vectorized[:len_train]
        y_train = entire_intents[:len_train]

        X_val = X_entire_vectorized[len_train:(len_train+len_val)]
        y_val = entire_intents[len_train:(len_train+len_val)]

        X_test = X_entire_vectorized[(len_train+len_val):]
        y_test = entire_intents[(len_train+len_val):]
        
        return X_train,y_train, X_val, y_val, X_test, y_test
    
    def fit_classifier(self, label, clf, X_train, y_train):
        print('======================================================')
        print("Training Classifier : ", label)
        return clf.fit(X_train, y_train)
    
    def evaluate_classifier(self, label, clf, X_test, y_test):
        print('======================================================')
        print("Testing Classifier: ", label)

        y_pred = clf.predict(X_test)
        #classification_report = classification_report(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision,recall,fscore,support=precision_recall_fscore_support(y_test,y_pred,average='macro')
        print('\nClassification Metrics:')
        print('Precision : {}'.format(precision))
        print( 'Recall    : {}'.format(recall))
        print('F-score   : {}'.format(fscore))
        print('Accuracy  : {}'.format(accuracy))
        print('======================================================')
 

def main():
    data_file = 'data_full.json'
    intentClassify = IntentClassifier(data_file, classifier_labels=["NaiveBayes"])
    
    entire_df, len_train, len_val, len_test = intentClassify.get_dataframe()
    entire_sents = list(entire_df["query"])
    entire_intents = list(entire_df["intent"])
        
    cleaned_entire_sents = intentClassify.cleaning(entire_sents)
    
    X_entire = intentClassify.vectorize_data(cleaned_entire_sents)
    
    X_train,y_train, X_val, y_val, X_test, y_test = intentClassify.split_data(X_entire, entire_intents, len_train, len_val, len_test)
    
    print('======================================================')
    
    if intentClassify.train:
        for label, clf in intentClassify.classifiers.items():
            trained_classifier = intentClassify.fit_classifier(label, clf, X_train, y_train)
            intentClassify.classifiers[label] = trained_classifier
    
    print('======================================================')
    
    if intentClassify.eval:
        for label, clf in intentClassify.classifiers.items():
            intentClassify.evaluate_classifier(label, clf, X_test, y_test)
      
main()