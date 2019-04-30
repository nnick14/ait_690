## Assignement 4 - Information Retrieval
# AIT 690
# Group Members: Jonathan Chiang, Nick Newman, Tyson Van Patten

# This program builds on the ir-system.py
# Given an input (the output file with the queries and relevant documents
# from the ir-system.py program) and a test set (cranqrel.py) the program will
# calculate the overall precision and recall of the program.
# There is no necessary input for this file, just make sure that the
# file is run in the same directory as the cran-output.txt and cranqrel files


## importing the necessary packages
from collections import defaultdict
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
#Read in files
our_cran = ''
with open("cran-output.txt") as myfile:
    our_cran = myfile.read()
solutions = ''
with open('cranqrel') as myfile:
    solutions = myfile.read()
#create dictionary of solutions from cranqrel
sol_dict = defaultdict(list)
for l in solutions.split('\n'):
    row = l.split(' ')
    if int(row[0]) in sol_dict.keys():
        sol_dict[int(row[0])].append(int(row[1]))
    else:
        sol_dict[int(row[0])].append(int(row[1]))
#create a dataframe of results from our cos_sim and the truth values
logger = open('mylogfile.txt','w')
ourcran1, ourcran2 = our_cran.split('\n----------------\n')



def scorer(ourcran, run_num):
    cran_pd = pd.DataFrame(columns=['query','document','source_val','our_val'])


    for l in ourcran.split('\n'):
        if(len(l.split(' '))>1):
            q, v = l.split(' ')
            source_val = 0
            our_val = 0
            #assign truth values
            if int(q) in sol_dict.keys():
                if int(v) in sol_dict[int(q)]:
                    source_val = 1
                    our_val = 1
                else:
                    source_val = 0
                    our_val = 1
            else:
                our_val = 0
            cran_pd.loc[len(cran_pd['query'])] = [int(q),int(v), source_val, our_val]
    #add in false negatives
    for k, v in sol_dict.items():
        for doc in v:
            found = cran_pd.loc[(cran_pd['query'] == k) & (cran_pd['document'] == doc)]
            if found.empty:
                cran_pd.loc[len(cran_pd['query'])] = [k, doc, 1, 0]
    #calculate precision and recall
    true_pos = 0
    false_pos = 0
    false_neg = 0
    for k, v in cran_pd.iterrows():
        if v['source_val'] == 1 and v['our_val'] == 1:
            true_pos += 1
        elif v['source_val'] == 0 and v['our_val'] == 1:
            false_pos += 1
        elif v['source_val'] == 1 and v['our_val'] == 0:
            false_neg += 1

    prec = true_pos/(true_pos + false_pos)
    rec = true_pos/(true_pos + false_neg)


    return [run_num, prec, rec]

def main():

    pool = ThreadPool(2)
    a = [ourcran1, ourcran2]
    b = [1,2]
    if len(ourcran1) > 0 and len(ourcran2)> 0:
        results = pool.starmap(scorer, zip(a,b))
        pool.close()
        pool.join()

        for r in results:
            run_num = r[0]
            prec = r[1]
            rec = r[2]
            if run_num == 2:

                print('Bigram Model Score:')
                logger.write('Bigram Model Score:\n')
            else:
                logger.write('Bag of Words Score:\n')
                print('Bag of Words Score:')
            print('Precision = ' + str(prec))
            print('Recall = ' + str(rec))
            logger.write('Precision = ' + str(prec) + '\n')
            logger.write('Recall = ' + str(rec) + '\n')

if __name__ == '__main__':
    main()

logger.close()
