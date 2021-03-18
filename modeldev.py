import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import pickle Package
import pickle
#from google.colab import files
Accounts = pd.read_csv("C:\\Users\\Donatus\\Documents\\AccountFraudDetect\\PS_20174392719_1491204439457.csv")
Accounts.head()
# Import label encoder 
from sklearn import preprocessing   
label_encoder = preprocessing.LabelEncoder() 
# Encode labels in column 'species'. 
Accounts['type_encode']= label_encoder.fit_transform(Accounts['type']) 
Accounts['nameOrig_encode']= label_encoder.fit_transform(Accounts['nameOrig']) 
Accounts['nameDest_encode']= label_encoder.fit_transform(Accounts['nameDest']) 
#Accounts['transfer_amount_category_encode']= label_encoder.fit_transform(Accounts['transfer_amount_category'])

from sklearn.model_selection import train_test_split
X = Accounts[['type_encode','nameOrig_encode','nameDest_encode']]
y = Accounts["isFraud"]
X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size = 0.33, stratify=y ,random_state = 42)

from sklearn.ensemble import RandomForestClassifier 
# creating a RF classifier 
clf = RandomForestClassifier(n_estimators = 15)   
# Training the model on the training dataset 
# fit function is used to train the model using the training sets as parameters 
clf.fit(X_train, y_train)

# performing predictions on the test dataset 
y_pred = clf.predict(X_test) 
# metrics are used to find accuracy or error 
from sklearn import metrics   
# using metrics module for accuracy calculation 
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

Pkl_Filename = "Pickle_RL_Model.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(clf, file)