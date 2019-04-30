## Assignement 4 - Information Retrieval
# AIT 690
# Group Members: Jonathan Chiang, Nick Newman, Tyson Van Patten

# This is a program that performs information retrieval. This means that given
# a query the program will return the ranked order of documents that are most
# relevant to the query. Given the query inputs, the output will show the query
# numbers, along with the document numbers, so you can easily see which ones
# are related. There is no necessary input for this file, just make sure that the
# file is run in the same directory as the cran.qry and cranqrel files
# the output_file will contain the matching queries and documents and will be fed
# into the next file: precision-recall.py

# We noticed that the traditional bag of words did not work as well as we would've
# liked, so we decided to implement bigrams in order to attempt to increase performance.
# This worked and increased the precision and recall values over using unigrams.
# However, we also believe that bigrams might not be able to account
# for the differences in how people phrase questions vs. how people phrase statements.


## importing the necessary packages
import re
from collections import defaultdict
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

## Running without bigrams
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
###stop words
stops = stopwords.words('english')
###question words are also stop words (for the query)
stops.extend(['who','what','when','where','how','why','.',',','?'])
stop_words = set(stops)
lemmatizer = WordNetLemmatizer()

with open("cran.qry") as myfile:
    text = myfile.read()

text = re.split(".I|.W",text)
text = [row.replace("\n"," ") for row in text]
text = [row.replace("  "," ") for row in text]
text = list(filter(None, text))

query_IDs = []
query_text = []
for i in range(0,len(text)):
    if i % 2 == 0:
        query_IDs.append(int(text[i]))
    else:
        query_text.append(text[i])

query_text2 = []
for row in query_text:
    sent = []
    for word in word_tokenize(row):
        word = re.sub(r'[^\w\s]',' ',word)
        word = re.sub('  ',' ',word)
        word = re.sub('   ',' ',word)
        sent.append(word)
    query_text2.append(sent)
query_text2 = [" ".join(x) for x in query_text2]

queries = dict(zip(query_IDs, query_text2))
queries_df = pd.DataFrame({'ID':query_IDs,'Text':query_text2})


## Reading in the document information and putting it into a query
with open("cran.all.1400") as myfile:
    text = myfile.read()
text = text.split(".I")

text = [row.replace("\n"," ") for row in text]
text = [row.replace("  "," ") for row in text]
text = list(filter(None, text))

chunks = []
for row in text:
    row = re.split(".I|.T|.A|.B|.W",row)
    chunks.append(row)

document_IDs = []
doc_text = []
doc_author = []
doc_title = []
doc_pub = []

count = 0
for row in chunks:
    document_IDs.append(int(row[0]))
    doc_title.append(row[1])
    doc_author.append(row[2])
    doc_pub.append(row[3])
    doc_text.append(row[4])

doc_text2 = []
for row in doc_text:
    sent = []
    for word in word_tokenize(row):
        word = re.sub(r'[^\w\s]',' ',word)
        word = re.sub('  ',' ',word)
        word = re.sub('   ',' ',word)
        sent.append(word)
    doc_text2.append(sent)


doc_text2 = [" ".join(x) for x in doc_text2]
documents = dict(zip(document_IDs, zip(doc_title, doc_author, doc_pub, doc_text2)))
documents_info = {'ID':document_IDs,'Title':doc_title,'Author':doc_author,'Publication':doc_pub,
                'Text':doc_text2}
documents_df = pd.DataFrame(documents_info)

## reading in the solution information and putting it into a dictionary
with open("cranqrel") as myfile:
    text = myfile.read()

text = text.split("\n")
chunks = []
for row in text:
    chunks.append(row.split(" "))

## creating solutions dictionary where key is the query ID and the values are the
## document IDs

#getting the tfidf for the documents
def doc_tfidf(document):
    ##getting the term frequency counts for each word in each document
    doc_tf_by_sent = []
    for row in document:
        tf = defaultdict(int)
        for word in word_tokenize(row):
            if word not in stop_words:
                tf[word] += 1
        doc_tf_by_sent.append(tf)

    #document idf
    num_docs = len(doc_text2)
    term_docs = defaultdict(int)
    for sent in doc_tf_by_sent:
        for word in sent:
            term_docs[word] += 1

    # idf for the document text
    idf = defaultdict(int)
    for k,v in term_docs.items():
        idf[k] = np.log(num_docs / (v+1))

    doc_tfidf_sent = []
    for sent in doc_tf_by_sent:
        tfidf = defaultdict(int)
        for k,v in sent.items():
            tfidf[k] = v * idf[k]
        doc_tfidf_sent.append(tfidf)

    return term_docs, idf, doc_tfidf_sent


term_docs, idf, doc_tfidf_sent = doc_tfidf(doc_text2)

#getting the tfidf for the queries
def query_tfidf(query, word_list, idf):

    ## getting the term frequency counts for each word in each query if the word
    ## is in the documents
    query_tf_by_sent = []
    for row in query:
        tf = defaultdict(int)
        for word in word_tokenize(row):
            if word in word_list:
                if word not in stop_words:
                    tf[word] += 1
        query_tf_by_sent.append(tf)

    ## tfidf query
    query_tfidf_sent = []
    for sent in query_tf_by_sent:
        tfidf = defaultdict(int)
        for k,v in sent.items():
            tfidf[k] = v * idf[k]
        query_tfidf_sent.append(tfidf)

    return query_tfidf_sent

query_tfidf_sent = query_tfidf(query_text2, term_docs, idf)


### tfidf for document text
def computetfidf(text, terms, tfidf):
    sorted_words = sorted(terms.keys())
    tfidfvector = np.empty([len(text), len(terms)])
    for i in range(len(text)):
        for j, word in enumerate(sorted_words):
            if word in text[i]:
                tfidfvector[i,j] = tfidf[i][word]
    return tfidfvector

### tfidf for document text
vect_doc = computetfidf(doc_text2, term_docs, doc_tfidf_sent)

### tfidf vect for query_text
vect_query = computetfidf(query_text2, term_docs, query_tfidf_sent)

### cosine similarity
# rows are documents
# columns are queries
def cos_sim(x,y):
    dot = np.dot(x,y.T)
    norma = np.sqrt((x*x).sum(axis=1))
    normb = np.sqrt((y*y).sum(axis=1))
    cos = dot/np.einsum('i,j',norma, normb)
    cos = np.nan_to_num(cos)
    return cos

cos = cos_sim(vect_doc, vect_query)

#ID is the query ID in cran.qry and the ID in cranqrel
#the first number in Q is the actual Query number and the second number is
#the number from cranqrel
cos_col = []
for i in range(len(query_IDs)):
    cos_col.append(str(i+1))

cos_row = []
for i in document_IDs:
    cos_row.append(str(i))

cos = pd.DataFrame(cos, index = cos_row,columns=cos_col)
logger = open('cran-output.txt','w')
output = []
# getting the cos values over .10 for each query
for row in cos.iteritems():
    #print(row[0])
    for k, v in row[1].sort_values(ascending=False).iteritems():
        if v > 0.15:
            #print(row[0],k,v)
            output.append(str(row[0]) + ' ' + str(k) + '\n')

for o in range(len(output)):
    if output[o] != output[-1]:
        logger.write(output[o])
    else:
        logger.write(output[o].replace('\n',''))


## implementation with bigrams
###lemmatize, remove stop words, and convert to bigrams
def tokens_to_bigrams(token_list):
    lem_list = []
    for t in token_list:
        if t not in stop_words:
            lem_list.append(lemmatizer.lemmatize(t))

    bigrams=ngrams(lem_list,2)
    return list(bigrams)

## Reading in the query information and putting it into a dictionary
with open("cran.qry") as myfile:
    text = myfile.read()

text = re.split(".I|.W",text)
text = [row.replace("\n"," ") for row in text]
text = [row.replace("  "," ") for row in text]
text = list(filter(None, text))

query_IDs = []
query_text = []
for i in range(0,len(text)):
    if i % 2 == 0:
        query_IDs.append(int(text[i]))
    else:
        query_text.append(text[i])

query_text2 = []
for row in query_text:
    sent = []
    for word in word_tokenize(row):
        word = re.sub(r'[^\w\s]',' ',word)
        word = re.sub('  ',' ',word)
        word = re.sub('   ',' ',word)
        sent.append(word)
    query_text2.append(sent)
query_text2 = [" ".join(x) for x in query_text2]
for q in range(len(query_text2)):
    query_text2[q] = tokens_to_bigrams(query_text[q].split(' '))

queries = dict(zip(query_IDs, query_text2))
queries_df = pd.DataFrame({'ID':query_IDs,'Text':query_text2})


## Reading in the document information and putting it into a query
with open("cran.all.1400") as myfile:
    text = myfile.read()
text = text.split(".I")

text = [row.replace("\n"," ") for row in text]
text = [row.replace("  "," ") for row in text]
text = list(filter(None, text))

chunks = []
for row in text:
    row = re.split(".I|.T|.A|.B|.W",row)
    chunks.append(row)

document_IDs = []
doc_text = []
doc_author = []
doc_title = []
doc_pub = []

count = 0
for row in chunks:
    document_IDs.append(int(row[0]))
    doc_title.append(row[1])
    doc_author.append(row[2])
    doc_pub.append(row[3])
    doc_text.append(row[4])

doc_text2 = []
for row in doc_text:
    sent = []
    for word in word_tokenize(row):
        word = re.sub(r'[^\w\s]',' ',word)
        word = re.sub('  ',' ',word)
        word = re.sub('   ',' ',word)
        sent.append(word)
    doc_text2.append(sent)


doc_text2 = [" ".join(x) for x in doc_text2]
for d in range(len(doc_text2)):
    doc_text2[d] = tokens_to_bigrams(doc_text2[d].split(' '))

documents = dict(zip(document_IDs, zip(doc_title, doc_author, doc_pub, doc_text2)))
documents_info = {'ID':document_IDs,'Title':doc_title,'Author':doc_author,'Publication':doc_pub,
                'Text':doc_text2}
documents_df = pd.DataFrame(documents_info)




#getting the tfidf for the documents
def doc_tfidf2(document):
    ##getting the term frequency counts for each word in each document
    doc_tf_by_sent = []
    for row in document:
        tf = defaultdict(int)
        for word in row:
                tf[word] += 1
        doc_tf_by_sent.append(tf)

    #document idf
    num_docs = len(doc_text2)
    term_docs = defaultdict(int)
    for sent in doc_tf_by_sent:
        for word in sent:
            term_docs[word] += 1

    # idf for the document text
    idf = defaultdict(int)
    for k,v in term_docs.items():
        idf[k] = np.log(num_docs / (v+1))

    doc_tfidf_sent = []
    for sent in doc_tf_by_sent:
        tfidf = defaultdict(int)
        for k,v in sent.items():
            tfidf[k] = v * idf[k]
        doc_tfidf_sent.append(tfidf)

    return term_docs, idf, doc_tfidf_sent


term_docs, idf, doc_tfidf_sent = doc_tfidf2(doc_text2)

#getting the tfidf for the queries
def query_tfidf2(query, word_list, idf):

    ## getting the term frequency counts for each word in each query if the word
    ## is in the documents
    query_tf_by_sent = []
    for row in query:
        tf = defaultdict(int)
        for word in row:
            if word in word_list:
                if word not in stop_words:
                    tf[word] += 1
        query_tf_by_sent.append(tf)

    ## tfidf query
    query_tfidf_sent = []
    for sent in query_tf_by_sent:
        tfidf = defaultdict(int)
        for k,v in sent.items():
            tfidf[k] = v * idf[k]
        query_tfidf_sent.append(tfidf)

    return query_tfidf_sent

query_tfidf_sent = query_tfidf2(query_text2, term_docs, idf)


### tfidf for document text
def computetfidf2(text, terms, tfidf):
    sorted_words = sorted(terms.keys())
    tfidfvector = np.empty([len(text), len(terms)])
    for i in range(len(text)):
        for j, word in enumerate(sorted_words):
            if word in text[i]:
                tfidfvector[i,j] = tfidf[i][word]
    return tfidfvector

### tfidf for document text
vect_doc = computetfidf2(doc_text2, term_docs, doc_tfidf_sent)

### tfidf vect for query_text
vect_query = computetfidf2(query_text2, term_docs, query_tfidf_sent)

###calculate cosine simularity.
cos = cos_sim(vect_doc, vect_query)

#ID is the query ID in cran.qry and the ID in cranqrel
#the first number in Q is the actual Query number and the second number is
#the number from cranqrel
cos_col = []
for i in range(len(query_IDs)):
    cos_col.append(str(i+1))

cos_row = []
for i in document_IDs:
    cos_row.append(str(i))

cos2 = pd.DataFrame(cos, index = cos_row,columns=cos_col)
output = []
# getting the cos values over .10 for each query
for row in cos2.iteritems():
    #print(row[0])
    for k, v in row[1].sort_values(ascending=False).iteritems():
        if v > 0.15:
            output.append(str(row[0]) + ' ' + str(k) + '\n')

logger.write('\n----------------\n')
for o in range(len(output)):
    if output[o] != output[-1]:
        logger.write(output[o])
    else:
        logger.write(output[o].replace('\n',''))

logger.close()
