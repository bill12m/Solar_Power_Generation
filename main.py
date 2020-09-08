import subprocess as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model

sp.call('clear', shell = True)

#Import datasets for Plant 1
plant_1_gen = pd.read_csv('data/Plant_1_Generation_Data.csv')
plant_1_weather = pd.read_csv('data/Plant_1_Weather_Sensor_Data.csv')

#Drop unneeded columns
plant_1_gen = plant_1_gen.drop(columns = ['PLANT_ID','TOTAL_YIELD'], axis = 1)
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
#eod_timestamp = plant_1_gen.where(plant_1_gen['DATE_TIME_AS_STRING'].str.contains('23:45:00')).fillna(method = 'backfill')
#eod_timestamp = eod_timestamp.sort_values(['SOURCE_KEY', 'DAT E_TIME'])
#ax = sns.lineplot(data = eod_timestamp, x = 'DATE_TIME', y = 'DAILY_YIELD',
#                  hue = 'SOURCE_KEY')

plant_1_gen = plant_1_gen.set_index('DATE_TIME').sort_values(['SOURCE_KEY', 'DATE_TIME'])

eod_timestamp = plant_1_gen.at_time('23:45:00')
for inverter in plant_1_gen['SOURCE_KEY'].unique():
    df = plant_1_gen[plant_1_gen['SOURCE_KEY'] == inverter].asfreq('15T')
    timestamp = df.at_time('23:45:00').fillna(method = 'backfill')
    differences = eod_timestamp.compare(timestamp, align_axis = 0)
    eod_timestamp.append(differences)
eod_timestamp = eod_timestamp.drop_duplicates()
#daily_sum = df_train[['DC_POWER','AC_POWER']].resample('1D').sum()

