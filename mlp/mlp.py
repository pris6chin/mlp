import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as pplt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import xgboost as xgb

#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
#from sklearn.preprocessing import LabelEncoder
#from sklearn.svm import SVC
#from sklearn.preprocessing import StandardScaler

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
selection_model_cols_iv= ['relative-humidity','windspeed','psi','temperature_trf']
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
dummy=pd.get_dummies(df[categorical_cols].astype(str),drop_first=True,dtype='uint8')
for i,v in enumerate(dummy.columns):
    df_model[v]=dummy[v]
    selection_model_cols_iv.append(v)
#print(df_model.columns)
#print(len(df_model.columns))
################################
#DOWNCASTING MY NUMBERS
down_cast_int = df_model.select_dtypes(include = 'integer').columns 
down_cast_float = df_model.select_dtypes(include = 'float').columns 

for i,v in enumerate(down_cast_int):
    df_model[v]=pd.to_numeric(df_model[v], downcast='unsigned')
for i,v in enumerate(down_cast_float):
    df_model[v]=pd.to_numeric(df_model[v], downcast='float')

print(df_model.info())

################################
#array = df_model.values
X = df_model[selection_model_cols_iv].values
y = df_model[selection_model_cols_dv].values

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size = 0.25, random_state=0)
#LinearRegression().fit(X_train, y_train)
# print(np.info(X_train), np.info(X_test), np.info(y_train), np.info(y_test))
# prepare configuration for cross validation test harness
X_fit= MinMaxScaler().fit_transform(X)
y_fit= MinMaxScaler().fit_transform(y)
kf = KFold(shuffle=True, n_splits=5)
# prepare models

X_train_fit= MinMaxScaler().fit_transform(X_train)
y_train_fit= MinMaxScaler().fit_transform(y_train)

X_test_fit= MinMaxScaler().fit_transform(X_test)
y_test_fit= MinMaxScaler().fit_transform(y_test)

pca = PCA(0.9)
X_fit_new = pca.fit_transform(X_fit)

print(pca.explained_variance_ratio_)
print(np.info(X_fit_new))


print('''------------------
Linear regression accuracy scores...''')
LR_accuracies = cross_val_score(estimator = LinearRegression(), X = X_fit_new, y = y_fit, cv = kf)
print(LR_accuracies.mean())
print(LR_accuracies.std())


print('''------------------
SVR accuracy scores...''')
SVR_accuracies = cross_val_score(estimator = SVR(kernel = 'rbf'), X = X_fit_new, y = y_fit.ravel(), cv = kf)
print(SVR_accuracies.mean())
print(SVR_accuracies.std())


print('''------------------
Decision Tree regression accuracy scores...''')
DTR_accuracies = cross_val_score(estimator = DecisionTreeRegressor(random_state = 0), X = X_fit_new, y = y_fit.ravel(), cv = kf)
print(DTR_accuracies.mean())
print(DTR_accuracies.std())

print('''------------------
Random Forest regression accuracy scores...''')
RFR_accuracies = cross_val_score(estimator = RandomForestRegressor(n_estimators = 10, random_state = 0), X = X_fit_new, y = y_fit.ravel(), cv = kf)
print(RFR_accuracies.mean())
print(RFR_accuracies.std())

print('''------------------
XGB accuracy scores...''')
XGB_accuracies = cross_val_score(estimator = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 200), X = X_fit_new, y = y_fit.ravel(), cv = kf)
print(XGB_accuracies.mean())
print(XGB_accuracies.std())
