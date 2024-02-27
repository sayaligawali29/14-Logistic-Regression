# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:32:17 2024

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import classification_report
claimants=pd.read_csv("claimants.csv")
#There are CLMAGE and loss are having continuons data rest
#verify the dataset where CASENUM is ot really useful so drop
c1=claimants.drop('CASENUM',axis=1)
c1.head(11)
c1.describe()
#let us check whether there are null values
c1.isna().sum()
#There are several null values
#if we will used dropna() function we will lose 200 data points
#hence we will go for imputation
c1.dtypes
mean_value=c1.CLMAGE.mean()
#hence all null values of CLMAGE has been filled by men value
#for columns where there are discrete values.we will apply mode
mode_CLMSEX=c1.CLMSEX.mode()
mode_CLMSEX
c1.CLMSEX=c1.CLMSEX.fillna((mode_CLMSEX)[0])
c1.CLMSEX.isna().sum()
#CLMINSUR is also categorical data hence mode imputation is applies
mode_CLMINSUR=c1.CLMINSUR.mode()
mode_CLMINSUR
c1.CLMINSUR=c1.CLMINSUR.fillna((mode_CLMINSUR)[0])
c1.CLMINSUR.isna().sum()
#SEATBELT is categorical data hence go for mode imputation
mode_SEATBELT=c1.SEATBELT.mode()
mode_SEATBELT
c1.SEATBELT=c1.SEATBELT.fillna((mode_SEATBELT)[0])
c1.SEATBELT.isna().sum()
#Now the person we met an accident will hire the atternev or not
#let us build  the model
logit_model=sm.logit('ATTORNEY ~CLMAGE+LOSS+CLMINSUR+CLMSEX+SEATBELT',data=c1).ftt()
logit_model.summary()
#in logistic regression we do not have R squared values only check p=value
#SEATBELT is statistically insignificant ignore and proceed
logit_model.summary2()
 #it is going to check AIC value it stand for akaike information criterion
 # is mathematival method for evaluation how well a model fits the data
 # A lower the score more the better model,AIC scores are only useful in
 #with other AIC scores for the same dataset
 
 #Now let us go for prediction
pred=logit_model.predict(c1.iloc[:,1:])
#hre we are applying all rows columns from 1, as column 0 is ATTORNEY
#target value
#let us check the performance of the model
fpr,tpr,thresholds=roc_curve(c1.ATTORNEY,pred)
#WE ARE APPLYING ACTUAL VALUES AND PREDICTED VALUES AS TO GET
#false positive rate true positive rate and threshold
#the optimal cutoff value is the point where there is high true positive
#you can use the below code to get the value:
optimal_idx=np.argmax(tpr-fpr)
optimal_threshold=thresholds[optimal_idx]
optimal_threshold
#Roc :reciever operating characteristics curve in logistic regression are
#determining best cutoff/threshold value
import pylab as pl
i=np.arrange(len(tpr))#index for df
#here tpr is of 559,so it will create a scale from 0 to 558
roc=pd.DataFrame({'fpr':pd.Series(fpr,index=i),
                  'tpr':pd.Series(tpr,index=i),
                  '1-fpr':pd.Series(1-fpr,index=i),
                  'tf':pd.Series(tpr -(1-fpr),index=i),
                  'thresholds':pd.Series(thresholds,index=i)})

#we want to create a dataframe which comprises of column fpr
#tpr,1-fpr -(1-fpr=tf
#tpe-(1-fpr) is zero or near to zero is the optimal cut off point
#plot ROC curve
plt.plot(fpr,tpr)
plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc.iloc[(roc.tf-0).abs().argsort()[:1]]
roc_auc=auc(fpr,tpr)
print("Area under the curve:%f"% roc_curve)
#area is 0.7601

#tpr vs 1-fpr
#plot tpr vs 1-fpr
fig,ax=pl.subplots()
pl.plot(roc['tpr'],color='red')
pl.plot(roc['1-fpr'],color='blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('Receiver operating characteristics')
ax.set_xtricklabel([])
#the optimal cut off point is one where tpr is high n fpr is low
#the optimal cut off point is 0.317
#so anything above this can be labeled as 1 else 0
#You can see from thr output/chart that where TPR is crossing 1-FPR
#FPR is 36% and TPR is 63%
#
#filling all the cells with zeros
c1['pred']=np.zeros(1340)
c1.loc[pred>optimal_threshold."pred"]=1
#let us check the classification reprt
classification=classification_report(c1["pred"],c1["ATTORNEY"])
classification
#Splitting the data into train and test
train_data,test_data=train_test_split(c1,test_size=0.3)

#model building
model=sm.logit('ATTORNEY ~CLMAGE+LOSS_CLMINSUR+CLMSEX+SEATBELT',data=train_data).fit()
model.summary()
#o values are below the condtion of 0.05
#but SEATBELT has got statistically insignificant
model.summary2()

#AIC values is 1110.3782,AIC Score are useful in comparison with other
#lower the AIC score better the model

#let us go for prediction
test_pred=logit_model.predict(test_data)
#creating new column for storing predicted class of ATTORNEY
test_data["test_pred"]=np.zeros(402)
test_data.loc[test_pred>optimal_threshold,"test_pred"]=c1
#confusion matric
confusion_matrix=pd.crosstab(test_data.test_pred,test_data.ATTORNEY)
confusion_matrix
accuracy_test(142+151)/(402)
accuracy_test
#0.69,this is going to change with everytime when you run

#classification report
classification_test=classification_report(test_data["test_pred"],test_data)
classification_test

#accuracy=0.73

#ROC curve and AUC
fpr,tpr,threshold=metrics.roc_curve(test_data['ATTORNEY'],test_pred)
#plot ROC Curve

plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

#area under the curve
roc_auc_test=metrics.auc(fpr,tpr)
roc_auc_test

#prediction on train data

train_pred=logit_model.predict(train_data)
#creating new column for storing predicted class of ATTORNEY

train_data["train_pred"]=np.zeros(938)
train_data.loc[train_pred>optimal_threshold,"train_pred"]=c1
#confusion matrix
confusion_matrix=pd.crosstab(train_data.test_pred,train_data.ATTORNEY)
confusion_matrix
accuracy_train=(315+347)/(938)
accuracy_train


#classification report
classification_train=classification_report(train_data["train_pred"],train_data)
classification_test

#accuracy=0.69

#ROC curve and auc
fpr,tpr,threshold=metrics.roc_curve(train_data['ATTORNEY'],train_pred)
#plot ROC Curve

plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

#area under the curve 
roc_auc_train=metrics.auc(fpr,tpr)
roc_auc_train