# Final Project - Disaster Response Data from Twitter
# AIT 690
# Group Members: Jonathan Chiang, Nick Newman, Tyson Van Patten

# Our code consists of the following sections: a section establishing the 
# baseline model, a section tuning the parameters of the text model in order
# to improve our model, and a final model (Random Forest) that has been
# trained on the fully cleaned and transformed text



### Section 1: Baseline Model #########
### different baseline methods used: most common category and decision tree
## measuring the baseline without first cleaning the data
## reading in the training data
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import ast
from sklearn.model_selection import cross_val_predict
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import string
import re
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore',category=UserWarning)



train = pd.read_csv("full_data.csv", encoding="latin_1")

## One of the classes was misnamed and one of the classes was missing
train = train.rename(columns = {'class':'Class'})
train.Class.value_counts()
train.loc[train['Class'] == 'NOD', 'Class'] = 'OD'

train.isna().sum()
train.loc[train['Class'].isna(), 'Class'] = 'NR'
train.loc[train['Class'] == ' NR','Class'] = 'NR'
## splitting the data into X and y values for sklearn
X = train['tweet']
y = train['Class'] 

## encoding the categorical y labels
le = LabelEncoder()
## MED:0, NR:1, OD:2, PD:3, RES:4
y = le.fit_transform(y)

## majority class baseline: 'NR' ##############
prediction = np.zeros(2505,) + 1
accuracy = accuracy_score(y, prediction)
## Majority Class accuracy is around 75%
print("Majority Class Baseline Accuracy:\t {:.2f}%".format(accuracy*100))


## filtering out the stopwords, since tweets will have many that are irrelevant
stop_words = set(stopwords.words('english'))


## Decision Tree Baseline ####################
## training with the tfidf matrix of the data
DT_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words)),
        ('clf', DecisionTreeClassifier(random_state=1))])

scores = cross_val_score(DT_pipeline, X, y, cv=10, scoring="accuracy")
## Decision Tree baseline accuracy is around 86%
print("Decision Tree Baseline CV Accuracy:\t {:.2f}% +- {:.2f}%".format(scores.mean()*100, scores.std()*100))


final_predictions = cross_val_predict(DT_pipeline, X, y, cv=10)

results_df = pd.DataFrame({'actual':le.inverse_transform(y),'predicted':le.inverse_transform(final_predictions)})
dt_base_crosstab_results = pd.crosstab(results_df.actual, results_df.predicted, colnames=['actual'],
                               rownames=['predicted'])

print("\n")
pd.set_option('display.width',200)
pd.set_option('display.max_columns',35)
print("Baseline Confusion Matrix")
print(dt_base_crosstab_results)


wrong_class = train.loc[np.where(results_df['actual'] != results_df['predicted'])]
wrong_class['predicted'] = results_df['predicted'].loc[np.where(results_df['actual'] != results_df['predicted'])]
wrong_class = wrong_class.loc[:,('tweet','Class','predicted')]

print("\nBaseline incorrectly classified")   
print(wrong_class)


### Section 2: Working with the text data to improve our model ##################
train = pd.read_csv("full_data.csv", encoding="latin_1")

## One of the classes was misnamed and one of the classes was missing
train = train.rename(columns = {'class':'Class'})
train.Class.value_counts()
train.loc[train['Class'] == 'NOD', 'Class'] = 'OD'

train.isna().sum()
train.loc[train['Class'].isna(), 'Class'] = 'NR'
train.loc[train['Class'] == ' NR','Class'] = 'NR'

## splitting the data into X and y values for sklearn
X = train['tweet']
y = train['Class'] 

## encoding the categorical y labels
le = LabelEncoder()
## MED:0, NR:1, OD:2, PD:3, RES:4
y = le.fit_transform(y)

## filtering out the stopwords, since tweets will have many that are irrelevant
stop_words = set(stopwords.words('english'))
porter_stemmer = PorterStemmer()
def tokenizer_porter(text):
    return [porter_stemmer.stem(word) for word in word_tokenize(text)]

## Decision Tree Adjusted ####################
## training with the tfidf matrix of the data
DT_pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', DecisionTreeClassifier(random_state=1))])


param_grid = {'vect__ngram_range': [(1,1),(1,2),(1,3),(2,2),(3,3)],
                'vect__stop_words': [stop_words,None],
                'vect__tokenizer': [None, tokenizer_porter]}



gs_DT_tfidf = GridSearchCV(DT_pipeline, param_grid,
                           scoring='accuracy',
                           cv=10,
                           verbose=1,
                           n_jobs=-1)

# Skipping over fitting the best parameters in order to save time,
# since we already did this
#gs_DT_tfidf.fit(X,y)
#print('Best Parameters: ', gs_DT_tfidf.best_params_)

### Getting the CV accuracy on the optimal parameters
DT_pipeline = Pipeline([
        ('vect', TfidfVectorizer(stop_words=stop_words,
                                 tokenizer=tokenizer_porter)),
        ('clf', DecisionTreeClassifier(random_state=1))])

scores = cross_val_score(DT_pipeline, X, y, cv=10, scoring="accuracy")
## Decision Tree baseline accuracy is around 86%
print("\nAdjusted Decision Tree CV Accuracy:\t {:.2f}% +- {:.2f}%".format(scores.mean()*100, scores.std()*100))


# Printing out the information for the adjusted Decision Tree
final_predictions = cross_val_predict(DT_pipeline, X, y, cv=10)

results_df = pd.DataFrame({'actual':le.inverse_transform(y),'predicted':le.inverse_transform(final_predictions)})
adj_dt_crosstab_results = pd.crosstab(results_df.actual, results_df.predicted, colnames=['actual'],
                               rownames=['predicted'])

print("\n")
pd.set_option('display.width',200)
pd.set_option('display.max_columns',35)
print(adj_dt_crosstab_results)


wrong_class = train.loc[np.where(results_df['actual'] != results_df['predicted'])]
wrong_class['predicted'] = results_df['predicted'].loc[np.where(results_df['actual'] != results_df['predicted'])]
wrong_class = wrong_class.loc[:,('tweet','Class','predicted','no_harvey')]

wrong_class['token'] = wrong_class['tweet'].apply(lambda x: word_tokenize(x))
print("\nIncorrectly Classified Instances")
print(wrong_class)

rows = []
for row in wrong_class[['Class','predicted','no_harvey']].iterrows():
    r = row[1]
    for word in ast.literal_eval(r.no_harvey):
        rows.append((r.Class,r.predicted,word))

words = pd.DataFrame(rows, columns=['actual','predicted','word'])

words['word'] = words.word.str.lower()
counts = words.groupby(['actual','predicted']).word.value_counts().to_frame().rename(columns={'word':'num_words'})

total_counts = counts['num_words'].groupby(level=0).nlargest(10).reset_index(level=0, drop=True)
print("\nCounts of words grouped by actual tags")
print(total_counts)


###### Section 3: Final Model (Random Forest) ##############
# creating a mapping for contractions to be tokenized
contractions ={
        "'s":"is",
        "'re":"are",
        "'ve":"have",
        "n't":"not"
        }

# creating a class to clean the text to our specifications
class TextPreprocess(BaseEstimator, TransformerMixin):
    def __init__(self, remove_links=True):
        self.remove_links = remove_links
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if self.remove_links:
            X = X.apply(lambda a: re.sub('https?:\/\/\S+','web_link',a))
        # remove non ascii characters
        X = X.apply(lambda a: a.encode('ascii','ignore').decode('ascii'))
        # fixing contractions  
        X = X.apply(lambda r: [contractions[word] if word in contractions else word for word in word_tokenize(r.lower())]) 
        # removing stopwords
        stop_words = set(stopwords.words('english'))
        X = X.apply(lambda a: [t for t in (a) if t not in stop_words])
        # removing punctuation
        X = X.apply(lambda a: [t for t in a if t not in string.punctuation])
        # stemming the wordss
        porter_stemmer = PorterStemmer()
        X = X.apply(lambda r: [porter_stemmer.stem(word) for word in r])
        X = [" ".join(i) for i in X]
        return X

# running the full random forest pipeline
RF_pipeline = Pipeline([
        ('PreProcess', TextPreprocess()),
        ('vect', TfidfVectorizer()),
        ('clf', RandomForestClassifier(random_state=1, n_estimators=500))])
scores = cross_val_score(RF_pipeline, X, y, cv=10, scoring="accuracy")
print("\nRandom Forest CV Accuracy:\t {:.2f}% +- {:.2f}%".format(scores.mean()*100, scores.std()*100))

final_predictions = cross_val_predict(RF_pipeline, X, y, cv=10)
results_df = pd.DataFrame({'actual':le.inverse_transform(y),'predicted':le.inverse_transform(final_predictions)})
rf_crosstab_results = pd.crosstab(results_df.actual, results_df.predicted, colnames=['actual'],
                               rownames=['predicted'])

print("\n")
pd.set_option('display.width',200)
pd.set_option('display.max_columns',35)
print(rf_crosstab_results)

wrong_class = train.loc[np.where(results_df['actual'] != results_df['predicted'])]
wrong_class['clean_text'] = TextPreprocess().fit_transform(wrong_class['tweet'])
wrong_class['predicted'] = results_df['predicted'].loc[np.where(results_df['actual'] != results_df['predicted'])]
wrong_class = wrong_class.loc[:,('tweet','clean_text','Class','predicted',)]

wrong_class['token'] = wrong_class['tweet'].apply(lambda x: word_tokenize(x))
print("\nIncorrectly Classified Instances")
print(wrong_class)

rows = []
for row in wrong_class[['Class','predicted','clean_text']].iterrows():
    r = row[1]
    for word in word_tokenize(r.clean_text):
        rows.append((r.Class,r.predicted,word))

words = pd.DataFrame(rows, columns=['actual','predicted','word'])

# getting the word counts of the misclassified instances
words['word'] = words.word.str.lower()
counts = words.groupby(['actual','predicted']).word.value_counts().to_frame().rename(columns={'word':'num_words'})
total_counts = counts['num_words'].groupby(level=0).nlargest(10).reset_index(level=0, drop=True)
print("\nCounts of words grouped by actual tags")
print(total_counts)


# getting the word counts by class of all the words in the data
clean_train = pd.DataFrame(TextPreprocess().fit_transform(train['tweet']), columns=['tweet'])
clean_train['Class'] = train['Class']
rows = []
for row in clean_train[['Class','tweet']].iterrows():
    r = row[1]
    for word in word_tokenize(r.tweet):
        rows.append((r.Class, word))

actual_words = pd.DataFrame(rows, columns=['actual','word'])
actual_counts = actual_words.groupby(['actual']).word.value_counts().to_frame().rename(columns={'word':'num_words'})
actual_counts = actual_counts['num_words'].groupby(level=0).nlargest(5).reset_index(level=0, drop=True)


# output with all the data and all the incorrectly classified instances
wrong_class = wrong_class.drop(['token','clean_text'],axis=1)
all_output = pd.concat([train['tweet'], results_df], axis=1)
#all_output.to_csv('all_output.csv')
#wrong_class.to_csv('wrong_class.csv')


skf =StratifiedKFold(n_splits=10, random_state=1)
for train, test in skf.split(X,y):
    RF_pipeline.fit(X[train],y[train])
    pred = RF_pipeline.predict(X[test])
    precision, recall, fscore, support = precision_recall_fscore_support(y[test],pred)

pd.options.display.float_format = '{:.2f}'.format
final_scores = pd.DataFrame([precision, recall, fscore], columns=le.classes_.tolist(), 
                   index=['Precision','Recall','Fscore'])
final_scores  = final_scores.apply(lambda r: round(r*100,2).astype(str) + "%")

print('\nPrecision-Recall Results:')
print(final_scores)
