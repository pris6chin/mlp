import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as pplt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

plt.style.use('seaborn-whitegrid')
url = 'https://aisgaiap.blob.core.windows.net/aiap6-assessment-data/scooter_rental_data.csv'
df = pd.read_csv(url)
###GOTTA DOWNCAST LATER DONT FORGET

#CLEANING WEATHER
weather_dict = {'lear':'clear',
                'CLEAR':'clear',
                'loudy':'cloudy',
                'CLOUDY':'cloudy',
                'LIGHT SNOW/RAIN': 'light snow/rain',
               }
weather_clean=[]
for i in df["weather"]:
    if i in weather_dict:
        weather_clean.append(weather_dict[i])
    else:
        weather_clean.append(i)
df["weather"] = weather_clean
print(df.weather.value_counts())
print('''--------
Seems like the weather data is clean now.''')

#CLEANING NEGATIVE USER VALUES
guest_clean=[]
registered_clean=[]
for i in df['guest-users']:
    if i <0:
        guest_clean.append(0)
    else:
        guest_clean.append(i)
for j in df['registered-users']:
    if j <0:
        registered_clean.append(0)
    else:
        registered_clean.append(j)
df['registered-users']=registered_clean
df['guest-users']=guest_clean
print('''--------
Seems like the user data is clean now.''')

#DROP DUPLICATES
df['date_time'] = pd.to_datetime(df['date'] + ' ' + df['hr'].astype(str) + ':00:00')
df.drop_duplicates(subset ="date_time", keep = 'first', inplace=True)
print('''------------------
Duplicates dropped!''')

#CREATE MONTH AND DAY COLUMNS
df['month'] =df['date_time'].dt.month
df['day'] = df['date_time'].dt.day_name()
print('''------------------
Month and day columns created''')

#CREATING TOTAL USERS
df['total-users'] = df["guest-users"] + df["registered-users"]
print('''------------------
Total user column created''')

#CREATING TRANSFORMED VARIABLES
temperature_trf=[]
feels_like_temperature_trf=[]
for i in df["temperature"]:
    temperature_trf.append((i - 100)**2)
for j in df["feels-like-temperature"]:
    feels_like_temperature_trf.append((j - 135)**2)
df["temperature_trf"]=temperature_trf
df["feels-like-temperature_trf"]=feels_like_temperature_trf
print('''------------------
Transformed columns temperature and feels-like-temperature created''')

#CHECKING IF DATA LOOKS OK
#print('''------------------
#Some sanity check on the data...''')
#print(df.describe())
#print(df.info())

#SELECT MODEL DF 
print('''------------------
Some sanity check on the data...''')
selection_model_cols_iv= ['relative-humidity','windspeed','psi','temperature_trf','feels-like-temperature_trf']
selection_model_cols_dv= ['total-users']
df_model = df[selection_model_cols_iv + selection_model_cols_dv]
print(df_model.describe())
print(df_model.info())

#APPLY DUMMY ENCODING
'''le = LabelEncoder()
categorical_cols=['hr','month','day']
df_model[categorical_cols] = df_model[categorical_cols].apply(lambda col: le.fit_transform(col))
print(df_model.info())
'''
categorical_cols=['hr','month','day']
dummy=pd.get_dummies(df[categorical_cols].astype(str),drop_first=True)
for i,v in enumerate(dummy.columns):
    df_model[v]=dummy.iloc[i]
    selection_model_cols_iv.append(v)
print(df_model.columns)
print(len(df_model.columns))
################################

################################
#array = df_model.values
X = df_model[selection_model_cols_iv].values
Y = df_model[selection_model_cols_dv].values
# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()