import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as pplt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
url = 'https://aisgaiap.blob.core.windows.net/aiap6-assessment-data/scooter_rental_data.csv'
df = pd.read_csv(url)


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
df["weather_clean"] = weather_clean
print(df.weather_clean.value_counts())
print('''--------
Seems like the weather data is clean now.''')

#DROP DUPLICATES
df_clean= df.drop_duplicates(subset ="date_time", keep = 'first')
print('''------------------
Since all duplicates on date_time were all the duplicates there were, we know that date_time uniquely identifies a row.
Also duplicates are dropped and we now have clean data.''')

#CREATE MONTH AND DAY COLUMNS
df_clean['month'] = df_clean['date_time'].dt.month
df_clean['day'] = df_clean['date_time'].dt.weekday_name


#CREATING TOTAL USERS
print(f"I am now adding a new column total-users.")
totalusers = df_clean["guest-users"] + df_clean["registered-users"]
df_clean['total-users'] = totalusers
