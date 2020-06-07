# -*- Rush Alemu @ SIS -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv ('COVID-19_Severity4.csv')

y = df.iloc[:, -1].values

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from pandas import set_option
set_option('display.width', 100)
set_option('precision', 3)
description = df.describe()
correlations = df.corr(method='pearson')

label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Comorbid'] = label_encoder.fit_transform(df['Comorbid'])
df['CurrentSmoker'] = label_encoder.fit_transform(df['CurrentSmoker'])
df['RespiratoryRateGreaterThan24'] = label_encoder.fit_transform(df['RespiratoryRateGreaterThan24'])
df['TemperatureGreaterThan37'] = label_encoder.fit_transform(df['TemperatureGreaterThan37'])
df['GroundGlassOpacity'] = label_encoder.fit_transform(df['GroundGlassOpacity'])
df['Class'] = label_encoder.fit_transform(df['Class'])

from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
array = df.values
X = array[:,0:17]
Y = array[:,17]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
set_printoptions(precision=3)

from sklearn.preprocessing import StandardScaler
from numpy import set_printoptions
array = df.values
X = array[:,0:17]
Y = array[:,17]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
set_printoptions(precision=3)

from sklearn.preprocessing import Binarizer
from numpy import set_printoptions
array = df.values
X = array[:,0:17]
Y = array[:,17]
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
shuffle=True
set_printoptions(precision=3)

from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
array = df.values
X = array[:,0:17]
Y = array[:,17]
test = SelectKBest(score_func=chi2, k=16)
fit = test.fit(X, Y)
set_printoptions(precision=3)
features = fit.transform(X)

x=df.iloc[:,:-1]
y=df.Class

class_counts = df.groupby('Class').size()

from imblearn.over_sampling import SMOTE
smote = SMOTE()

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.20, random_state=101)

X_train_smote, y_train_smote = smote.fit_sample(x_train.astype('float'),y_train)

model= DecisionTreeClassifier()
tramodel=model.fit(x_train, y_train)
predict = tramodel.predict(x_test)

#saving model to disk
pickle.dump(tramodel, open('covid_sev_pre_model.pkl', 'wb'))

#loading model to compare the results
model = pickle.load(open('covid_sev_pre_model.pkl', 'rb'))




