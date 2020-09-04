import subprocess as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn import linear_model

#Pre-Made function with credit to 
#https://grisha.org/blog/2016/02/16/triple-exponential-smoothing-forecasting-part-ii/
def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # we are forecasting
          value = result[-1]
        else:
          value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result

def determine_best_alpha_beta(series,)

sp.call('clear', shell = True)

#Import datasets for Plant 1
plant_1_gen = pd.read_csv('data/Plant_1_Generation_Data.csv')
plant_1_weather = pd.read_csv('data/Plant_1_Weather_Sensor_Data.csv')

#Drop unneeded columns
plant_1_gen = plant_1_gen.drop(columns = ['PLANT_ID'], axis = 1)
plant_1_weather = plant_1_weather.drop(columns = ['PLANT_ID','SOURCE_KEY'], axis = 1)

#Convert DATE_TIME from object to timestamp
plant_1_gen['DATE_TIME'] = pd.to_datetime(plant_1_gen['DATE_TIME'], dayfirst = True)
plant_1_weather['DATE_TIME'] = pd.to_datetime(plant_1_weather['DATE_TIME'], dayfirst = True)
plant_1_gen['DATE_TIME_AS_STRING'] = plant_1_gen['DATE_TIME'].astype('string')
plant_1_weather['DATE_TIME_AS_STRING'] = plant_1_weather['DATE_TIME'].astype('string')

#Convert SOURCE_KEY from object to integer we can use
new_sourcekey_num = list(np.arange(0,22))
old_source_key = list(plant_1_gen['SOURCE_KEY'].unique())
for n in range(len(old_source_key)):
    plant_1_gen = plant_1_gen.replace(old_source_key[n],new_sourcekey_num[n])
del(old_source_key,new_sourcekey_num,n)

#Since DAILY_YIELD is a running total, filter plant_1_gen by the timestamp at
#the end of each day and then sort each inverter in chronological order.
eod_timestamp = plant_1_gen.where(plant_1_gen['DATE_TIME_AS_STRING'].str.contains('23:45:00')).dropna()
eod_timestamp = eod_timestamp.sort_values(['SOURCE_KEY', 'DATE_TIME'])
ax = sns.lineplot(data = eod_timestamp, x = 'DATE_TIME', y = 'DAILY_YIELD',
                  hue = 'SOURCE_KEY')

df = eod_timestamp.loc[eod_timestamp['SOURCE_KEY'] == 0, ['DAILY_YIELD']]
series = np.array(df['DAILY_YIELD'].values).tolist()
result = double_exponential_smoothing(series, alpha = 0.5, beta = 0.5)