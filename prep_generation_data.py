import pandas as pd
import numpy as np
import subprocess as sp

sp.call('clear', shell = True)

df = pd.read_csv('data/Plant_1_Generation_Data.csv').drop(columns = ['PLANT_ID', 'TOTAL_YIELD'], axis = 1)
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], dayfirst = True)
df = df.sort_values(['SOURCE_KEY', 'DATE_TIME']).set_index('DATE_TIME')

#Convert SOURCE_KEY from object to integer we can use
new_sourcekey_num = list(np.arange(0,22))
old_source_key = list(df['SOURCE_KEY'].unique())
for n in range(len(old_source_key)):
    df = df.replace(old_source_key[n],new_sourcekey_num[n])
del(old_source_key,new_sourcekey_num,n)

#Filter by source_key and fill in missing timestamps
filter_df_inverter = []
for inverter in df['SOURCE_KEY'].unique():
    df_inverter = df[df['SOURCE_KEY'] == inverter]
    ac = df_inverter['AC_POWER'].resample('1D').sum()
    dc = df_inverter['DC_POWER'].resample('1D').sum()
    daily_yield = df_inverter['DAILY_YIELD'].resample('1D').max()
    
    d = {'SOURCE_KEY':df_inverter['SOURCE_KEY'],
        'AC_POWER': ac,
         'DC_POWER': dc,
         'DAILY_YIELD': daily_yield}
    new_df = pd.DataFrame(data = d,
                          index = ac.index)
    filter_df_inverter.append(new_df)

#Create a new dataframe with the end-of-day yields for every inverter.
eod = filter_df_inverter[0].reset_index()
for inverter in range(1,df['SOURCE_KEY'].nunique()):
    df_inverter = filter_df_inverter[inverter].reset_index()
    eod = pd.concat([eod, df_inverter])
eod = eod.set_index('DATE_TIME').fillna(method = 'bfill')
del (inverter, df_inverter, filter_df_inverter, ac, dc)

eod.to_csv('data/Plant_1_End_of_Day.csv')
    
    