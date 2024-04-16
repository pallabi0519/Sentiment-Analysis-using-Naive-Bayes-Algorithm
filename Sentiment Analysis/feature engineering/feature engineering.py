import sys
import os
import nltk
import re
import math
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams



porter = PorterStemmer()
all_stopwords = stopwords.words('english')
#fetch datasets
train_data = sys.argv[1]
test_data = sys.argv[2]

pos_path = train_data + '/pos'
neg_path = train_data + "/neg"

#function for read data line by line
def read_text_file(file_path):
    #print(file_path)
    with open(file_path, 'r') as f:
        sentences = f.readlines()
        
    return str(sentences)

def get_dict(data,n):
    
    class_1=0
    gram_dictionary = {}

    for sentences in data:
        #Y_train.append(1)
        class_1 += 1
        clean = re.compile('<.*?>')
        sentences = re.sub(clean, '', sentences)
        #print(sentences)
        #filter_sentences = (sentences.translate(str.maketrans("","",string.punctuation)).lower()).split()
        filter_sentences = re.sub("[^\w]", " ",  sentences).split()
        sentence_without_sw = [word for word in filter_sentences if not word.lower() in all_stopwords]  #removing stopwords

        stemmed_sentences=[porter.stem(word) for word in sentence_without_sw]   #stemming all words

        gram_sentences = list(ngrams(stemmed_sentences, n)) 

        #n_pos.append(len(stemmed_sentences))

        for words in gram_sentences:
        #print(words)
            if words in gram_dictionary:
                gram_dictionary[words] += 1
            else:
                gram_dictionary[words] = 1
    
    return gram_dictionary, class_1
    

'''-------------------------processing the positive data----------------------------'''

X_train=[]
Y_train=[]
positive_sentences=[]
#os.chdir(pos_path)
#print(pos_path)
#print(neg_path)

for file in os.listdir(pos_path):
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{pos_path}/{file}"
  
        # call read text file function
        pos_sentences = read_text_file(file_path)
        #print(pos_sentences)
        positive_sentences.append(pos_sentences)
        X_train.append(pos_sentences)


bigram_pos_dictionary, Class1_bigram = get_dict(positive_sentences,2)
trigram_pos_dictionary, Class1_trigram = get_dict(positive_sentences,3)


'''-------------------------processing the negative data----------------------------'''

negative_sentences=[]
n_neg=[]
#os.chdir(pos_path)

for file in os.listdir(neg_path):
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f'{neg_path}/{file}'
  
        # call read text file function
        neg_sentences = read_text_file(file_path)
        #print(pos_sentences)
        negative_sentences.append(neg_sentences)
        X_train.append(neg_sentences)
#print(negative_sentences)


bigram_neg_dictionary, Class0_bigram = get_dict(negative_sentences,2)
trigram_neg_dictionary, Class0_trigram = get_dict(negative_sentences,3)




'''-------------------------processing the complete Training data----------------------------'''


bigram_dictionary, Class = get_dict(X_train,2)
trigram_dictionary, Class = get_dict(X_train,3)


'''---------------------fetching and processing test data----------------------'''

test_pos_path = test_data + '/pos'
test_neg_path = test_data + "/neg"

X_test=[]
Y_test=[]
neg_data=[]
pos_data=[]
for file in os.listdir(test_pos_path):
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{test_pos_path}/{file}"
  
        # call read text file function
        pos_sentences = read_text_file(file_path)
        #print(pos_sentences)
        pos_data.append(pos_sentences)
        X_test.append(pos_sentences)
for data in pos_data:
    Y_test.append(1)


for file in os.listdir(test_neg_path):
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{test_neg_path}/{file}"
  
        # call read text file function
        neg_sentences = read_text_file(file_path)
        
        neg_data.append(neg_sentences)
        X_test.append(neg_sentences)
for data in neg_data:
    Y_test.append(0)




'''----------------------Training the naive bayes model with after removing stopwords and applying stemming on data----------------------'''

def naiveClassifier(dictionary, pos_dict, neg_dict):
    
    #print(len(neg_dict))
    v=len(dictionary)
    phi_0={}
    phi_1={}
    c=1
    
    for word in dictionary.keys():
        if word in neg_dict:
                phi_0[word] = ((neg_dict[word]+c)/(len(neg_dict) + v*c))
        else:
                phi_0[word] = ((c) / (len(neg_dict) + v*c))
        if word in pos_dict:
                phi_1[word] = ((pos_dict[word]+c)/(len(pos_dict) + v*c))
        else:
                phi_1[word] = ((c)/(len(pos_dict) + v*c))
    return phi_0, phi_1

'''-----------------------------------------------bigram data--------------------------------------------'''
phi_0_bigram, phi_1_bigram = naiveClassifier(bigram_dictionary, bigram_pos_dictionary, bigram_neg_dictionary)


 
m=len(X_train)
p0=Class0_bigram/m
p1=Class1_bigram/m

#####function for predict the model and computing the accuracy of the model#####

def get_accuracy_bigram(review, actual, phi0, phi1):
    pred_class=[]
    #actual_class=[]
    
    for sentence in review:
        
        test_class_pos=test_class_neg=0
        clean = re.compile('<.*?>')
        sentences = re.sub(clean, '', sentence)
        filter_sentences = re.sub("[^\w]", " ",  sentences).split()
        sentence_without_sw = [word for word in filter_sentences if not word.lower() in all_stopwords]
        stemmed_sentences=[porter.stem(word) for word in sentence_without_sw]
        bigram_sentences = list(ngrams(stemmed_sentences, 2)) 

        for w in bigram_sentences:
            if w in phi0: test_class_neg += math.log(phi0[w])
            else: test_class_neg += math.log(1)
            if w in phi1: test_class_pos += math.log(phi1[w])
            else: test_class_pos += math.log(1)
        
           
        test_class_neg += math.log(p0)
        test_class_pos += math.log(p1)


        if (test_class_neg > test_class_pos): 
            pred_class.append(0)
        else: 
            pred_class.append(1)

    yes=no=0
    for i in range(len(pred_class)):
        if pred_class[i] == actual[i]:
            yes+=1
        else:
            no+=1
    accuracy = (yes/(yes+no))*100
    return accuracy, pred_class



test_accuracy_bigram,  test_pred_bigram = get_accuracy_bigram(X_test, Y_test, phi_0_bigram, phi_1_bigram)
print("Result (e) : The test accuracy of the model on bigram data is = {:2.3f}%".format(test_accuracy_bigram))

'''-------------------------------------------------trigram------------------------------------------------'''


phi_0_trigram, phi_1_trigram = naiveClassifier(trigram_dictionary, trigram_pos_dictionary, trigram_neg_dictionary)


 
m=len(X_train)
p0=Class0_trigram/m
p1=Class1_trigram/m

#####function for predict the model and computing the accuracy of the model#####

def get_accuracy_trigram(review, actual, phi0, phi1):
    pred_class=[]
    #actual_class=[]
    
    for sentence in review:
        
        test_class_pos=test_class_neg=0
        clean = re.compile('<.*?>')
        sentences = re.sub(clean, '', sentence)
        filter_sentences = re.sub("[^\w]", " ",  sentences).split()
        sentence_without_sw = [word for word in filter_sentences if not word.lower() in all_stopwords]
        stemmed_sentences=[porter.stem(word) for word in sentence_without_sw]
        trigram_sentences = list(ngrams(stemmed_sentences, 3)) 

        for w in trigram_sentences:
            if w in phi0: test_class_neg += math.log(phi0[w])
            else: test_class_neg += math.log(1)
            if w in phi1: test_class_pos += math.log(phi1[w])
            else: test_class_pos += math.log(1)
        
           
        test_class_neg += math.log(p0)
        test_class_pos += math.log(p1)


        if (test_class_neg > test_class_pos): 
            pred_class.append(0)
        else: 
            pred_class.append(1)

    yes=no=0
    for i in range(len(pred_class)):
        if pred_class[i] == actual[i]:
            yes+=1
        else:
            no+=1
    accuracy = (yes/(yes+no))*100
    return accuracy, pred_class



test_accuracy_trigram,  test_pred_trigram = get_accuracy_trigram(X_test, Y_test, phi_0_trigram, phi_1_trigram)
print("Result (a) : The test accuracy of the model on trigram data is = {:2.3f}%".format(test_accuracy_trigram))



###### finding confusion matrix ###### 
def create_confusion_matrix(actual_class, pred_class):
    numPred_l1=sum(pred_class)
    numPred_l0=len(pred_class)-numPred_l1

    numAct_l1=sum(actual_class)
    numAct_l0=len(actual_class)-numAct_l1

    Tn=Tp=0
    Fn=Fp=0
    for i in range(len(actual_class)):
        if (actual_class[i]==pred_class[i]) and (pred_class[i] == 0):
            Tn+=1
        elif(actual_class[i] == pred_class[i]) and (pred_class[i] == 1):
            Tp+=1
        elif(pred_class[i] == 0):
            Fn+=1
        else:
            Fp+=1

    return np.array([[Tn, Fp], [Fn, Tp]])

conf_matrix_bigram_test = create_confusion_matrix(Y_test, test_pred_bigram)
print('confusion matrix for bigram test data \n', conf_matrix_bigram_test)

conf_matrix_trigram_test = create_confusion_matrix(Y_test, test_pred_trigram)
print('confusion matrix for trigram test data \n', conf_matrix_trigram_test)
