import sys
import os
import numpy as np


test_data = sys.argv[2]

#function for read data line by line
def read_text_file(file_path):
    #print(file_path)
    with open(file_path, 'r') as f:
        sentences = f.readlines()
        
    return str(sentences)


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

'''-----------------------------------predicting with random target-----------------------------------'''

def check_test_random(x_test, y_test, numClasses):
    y_pred = np.random.randint(numClasses, size=len(y_test))
    correct = 0
    for i in range(len(x_test)):
        if y_pred[i] == y_test[i]:
            correct += 1
    acc = (correct / len(x_test))*100
    return acc, y_pred

test_accuracy_random, test_random_pred = check_test_random(X_test, Y_test,2)

print("Result (bi) : The test accuracy of the model by random predictions={:2.3f}%".format(test_accuracy_random))


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

print("Result (bii) : The test accuracy of the model by predicting all values positive={:2.3f}%".format(test_accuracy_pos))



'''
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


conf_matrix_random_test = create_confusion_matrix(Y_test, test_random_pred)

conf_matrix_pos_test = create_confusion_matrix(Y_test, test_pos_pred)

print('confusion matrix for random predict data \n', conf_matrix_random_test)
print('confusion matrix for positive predict data \n', conf_matrix_pos_test)
'''