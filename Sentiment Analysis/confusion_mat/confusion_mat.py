
import sys
import os

import re
import math
import numpy as np

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

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
        
        #words = re.sub("[^\w]", " ",  sentence).split()
    return str(sentences)


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
#print(positive_sentences)
n_pos =[]
class_pos=0
pos_dictionary = {}

for sentences in positive_sentences:
    Y_train.append(1)
    class_pos += 1
    clean = re.compile('<.*?>')
    sentences = re.sub(clean, '', sentences)
    #print(sentences)
    #filter_sentences = (sentences.translate(str.maketrans("","",string.punctuation)).lower()).split()
    filter_sentences = re.sub("[^\w]", " ",  sentences).split()
    #filter_sentences = [i.replace('br', '') for i in filter_sentences]
    #print(filter_sentences)
    #print(sentences[0])
    n_pos.append(len(filter_sentences))
    
    
    ###creating dictionary for positive class
    for words in filter_sentences:
        
        #print(words)
        if words in pos_dictionary:
            pos_dictionary[words] += 1
        else:
            pos_dictionary[words] = 1
    #print(pos_dictionary)

n_pos=sum(n_pos)
#print('pos',n_pos)

#print(len(pos_dictionary))

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

neg_dictionary = {}
class_neg=0
for sentences in negative_sentences:
    Y_train.append(0)
    class_neg += 1
    clean = re.compile('<.*?>')
    sentences = re.sub(clean, '', sentences)
    #print(sentences)
    #filter_sentences = (sentences.translate(str.maketrans('','',string.punctuation)).lower()).split()
    filter_sentences = re.sub("[^\w]", " ",  sentences).split()
    #print(filter_sentences)
    #print(sentences[0])
    #print(len(filter_sentences))
    
    n_neg.append(len(filter_sentences))

    ###creating dictionary for negative class
    for words in filter_sentences:
        
        if words in neg_dictionary:
            neg_dictionary[words] += 1
        else:
            neg_dictionary[words] = 1
            
n_neg=sum(n_neg)


'''-------------------------processing the complete Training data----------------------------'''

dictionary={}

for sentences in X_train:
    
    clean = re.compile('<.*?>')
    sentences = re.sub(clean, '', sentences)
    #print(sentences)
    #filter_sentences = (sentences.translate(str.maketrans('','',string.punctuation)).lower()).split()
    filter_sentences = re.sub("[^\w]", " ",  sentences).split()
    #print(filter_sentences)
    #print(sentences[0])
    #print(len(filter_sentences))

    ###creating dictionary for complete training data
    for words in filter_sentences:
        if words in dictionary:
            dictionary[words] += 1
        else:
            dictionary[words] = 1



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

'''------------------------------------Training the naive bayes model-----------------------------------'''

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


phi_0, phi_1 = naiveClassifier(dictionary, pos_dictionary, neg_dictionary)

m=len(X_train)
p0=class_neg/m
p1=class_pos/m


#####function for predict the model and computing the accuracy of the model

def get_accuracy(review, actual, phi0, phi1):
    pred_class=[]
    #actual_class=[]
    
    
    for sentence in review:
        
        test_class_pos=test_class_neg=0
        clean = re.compile('<.*?>')
        sentences = re.sub(clean, '', sentence)
        filter_sentences = re.sub("[^\w]", " ",  sentences).split()
        
        #actual_class.append(Y_train)
        
        
        for w in filter_sentences:
            if w in phi0: test_class_neg += math.log(phi0[w])
            else: test_class_neg += math.log(1)
            if w in phi1: test_class_pos += math.log(phi1[w])
            else: test_class_pos += math.log(1)
        
        test_class_neg += math.log(p0)
        test_class_pos += math.log(p1)
        #print('test0=',test_class_neg)
        #print('test1=',test_class_pos)
        
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

######computing and printing the accuracy for train data and test data

#train_accuracy, train_pred = get_accuracy(X_train, Y_train, phi_0, phi_1)
#print("Result (ai) : The train accuracy of the model on raw data is = {:2.3f}%".format(train_accuracy))


test_accuracy, test_pred = get_accuracy(X_test, Y_test, phi_0, phi_1)
#print("Result (ai) : The test accuracy of the model on raw data is = {:2.3f}%".format(test_accuracy))

'''-----------------------------------predicting with random target-----------------------------------'''

def check_test_random(x_test, y_test, numClasses):
    y_pred = np.random.randint(numClasses, size=len(x_test))
    correct = 0
    for i in range(len(x_test)):
        if y_pred[i] == y_test[i]:
            correct += 1
    acc = (correct / len(x_test))*100
    return acc, y_pred

test_accuracy_random, test_random_pred = check_test_random(X_test, Y_test,2)


'''----------------------------------------------------predicting as all positive sampples-----------------------------------------------------'''

def check_test_pos(x_test, y_test):
    y_pred = np.ones(len(y_test))
    correct = 0
    for i in range(len(x_test)):
        if y_pred[i] == y_test[i]:
            correct += 1
    acc = (correct / len(x_test))*100
    return acc, y_pred

test_accuracy_pos, test_pos_pred = check_test_pos(X_test, Y_test)


'''----------------------------------------Confusion matrix for part a & b-----------------------------------'''



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

conf_matrix_test = create_confusion_matrix(Y_test, test_pred)
print('confusion matrix for raw test data \n', conf_matrix_test)


conf_matrix_random_test = create_confusion_matrix(Y_test, test_random_pred)

conf_matrix_pos_test = create_confusion_matrix(Y_test, test_pos_pred)

print('confusion matrix for random predict data \n', conf_matrix_random_test)
print('confusion matrix for positive predict data \n', conf_matrix_pos_test)

