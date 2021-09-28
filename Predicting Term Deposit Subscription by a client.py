#Name    : Jayesh Dattatray Kulkarni
#Batch   : PGA-15
#Project : SVM

import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
# from imblearn.under_sampling import NearMiss
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.over_sampling import SMOTE

bank = pd.read_csv("bank-additional-full.csv",sep=';')

# Dimension of data set
bank.shape

bank.dtypes

# check Null
bank.isnull().sum()

# Check Zeros
bank[bank==0].count()

allcolumns = list(bank.columns)

# class count
class_count_0, class_count_1 = bank['y'].value_counts()

# Separate class
class_0 = bank[bank['y'] == 'no']
class_1 = bank[bank['y'] == 'yes']

class_0_under = class_0.sample(class_count_1)

bank_under = pd.concat([class_0_under, class_1], axis=0)

allcolumns = list(bank_under.columns)
numColumn = list(bank_under.select_dtypes(include=['int32','int64','float32','float64']).columns.values)
itrnum = 0
itrfact = 0
for i in allcolumns: 
    countList = list(bank_under[i].value_counts())
    if (round(countList[0]/len(bank_under), 2) >= 0.85):
        if(i in numColumn):
            print(i)
            itrnum = itrnum + 1
            bank_under = bank_under.drop(i, axis=1)
        else:
            print(i)
            itrfact = itrfact + 1
            bank_under = bank_under.drop(i, axis=1)
    
print(f'Numeric Columns are droped from dataset after checking singularity Total Columns = {itrnum}')
print(f'Factor Columns are droped from dataset after checking singularity Total Columns = {itrfact}')


factorColumn = list(bank_under.select_dtypes(include=['object']).columns.values)
factorColumn.remove('y')
for c in factorColumn:
    dummy = pd.get_dummies(bank_under[c],drop_first=True, prefix=c)
    bank_under = bank_under.join(dummy)

bank_under = bank_under.drop(factorColumn, axis=1)

le = LabelEncoder()
bank_under['target'] = le.fit_transform(bank_under['y'])
bank_under = bank_under.drop('y',axis=1)

bank_under = bank_under.sample(frac=1)

# standardize the dataset
bank_under_std = bank_under.copy()

minmax = preprocessing.MinMaxScaler()
bank_under_std.iloc[:,:] = minmax.fit_transform(bank_under_std.iloc[:,:])
bank_under_std['target'] = bank_under_std['target']
bank_under_std

trainx, testx, trainy, testy = train_test_split(bank_under_std.drop('target',axis=1), bank_under_std['target'], test_size=0.25)
print(f'trainx = {trainx.shape}, trainy = {trainy.shape}, testx = {testx.shape}, testy = {testy.shape}')

kernels = ['linear','rbf','poly','sigmoid']

# get the R-square for each kernel
for k in kernels:
    model = svm.SVR(kernel=k).fit(trainx,trainy)
    rsq = model.score(testx,testy)
    print(f'kernel = {k}, R_square = {rsq}')
    
m_mse = []
model = []

# 1) kernel = linear
svm_linear_model = svm.SVR(kernel='linear').fit(trainx,trainy)
model.append('linear')
svm_linear_prediction = svm_linear_model.predict(testx)
mse_linear_model = round(mean_squared_error(testy, svm_linear_prediction),3)
m_mse.append(mse_linear_model)

# plot the actual vs predicted Y (target)
ax1 = sns.distplot(testy, hist=False, color='r', label='Actual Y')
sns .distplot(svm_linear_prediction, hist=False, color='b', label='Predicted Y', ax=ax1)

# compare actual and predicted values
pd.DataFrame({'actual':testy,'predicted':np.round(svm_linear_prediction,2)}).head(10)