import os
import argparse
import numpy as np
from Classifier.clf import *
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset',help='Dataset',type =str,default=None,required=True)
parser.add_argument('-l','--label',type= str,default=None,required=True)

args= parser.parse_args()
print("reading the dataset")

df= pd.read_csv(args.dataset)
print("spliting the dataset into idv and dv")

x = df.drop(labels=args.label,axis=1)
y = df[args.label]

print("splitting the datset into train and test splits")
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=6)

clf = Classifier(train_x=x_train,train_y=y_train,test_x=x_test,test_y=y_test)

train_results = clf.train()
test_results = clf.test()
confusion_matrix_train = clf.confusion_matrix_train()
confusion_matrix_test= clf.confusion_matrix_test() 
classification_report_train = clf.classification_report_train()
classification_report_test = clf.classification_report_test()   
    
    
print("Exporting the train results...")
train_results.to_csv("train_results.csv",index=False)
print("Exporting the test results...")
test_results.to_csv("test_results.csv",index= False)

print("Exporting the confusion matrix for train...")

print(confusion_matrix_train)


print("Exporting the confusion matrix for test...")

print(confusion_matrix_test)


classification_report_train.to_csv("Classification_report_train")

classification_report_test.to_csv("Classification_report_test")



