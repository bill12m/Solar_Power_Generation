import subprocess as sp
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from datetime import time

class PrepGenerationData:
    def __init__(self,filename):
        self.filename = filename
    
    def prep_data(self):
        df = pd.read_csv(self.filename).drop(columns = ['PLANT_ID', 'TOTAL_YIELD'], axis = 1)
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], dayfirst = True)
        df = df.sort_values(['SOURCE_KEY', 'DATE_TIME']).set_index('DATE_TIME')
        df.iloc[df.index.indexer_between_time(time(0), time(4)), -1] = 0
        
        #Convert SOURCE_KEY from object to integer we can use
        new_sourcekey_num = list(np.arange(0,22))
        old_source_key = list(df['SOURCE_KEY'].unique())
        for n in range(len(old_source_key)):
            df = df.replace(old_source_key[n],new_sourcekey_num[n])
        del(old_source_key,new_sourcekey_num,n)
        
        df.iloc[df.index.indexer_between_time(time(0), time(4)), -1] = 0
            
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
        
        return(eod)
class PrepWeatherData:
    def __init__(self,filename):
        self.filename = filename
    
    def prep_data(self):
        df_weather = pd.read_csv(self.filename).drop(columns = 'PLANT_ID', axis = 1)
        df_weather['DATE_TIME'] = pd.to_datetime(df_weather['DATE_TIME'], dayfirst = True)
        df_weather = df_weather.set_index('DATE_TIME').asfreq('15T')
        df_weather = df_weather.interpolate()
        
        ambient = df_weather['AMBIENT_TEMPERATURE'].resample('1D').mean()
        module = df_weather['MODULE_TEMPERATURE'].resample('1D').mean()
        irradiation = df_weather['IRRADIATION'].resample('1D').mean()
        
        d = {'AMBIENT_TEMPERATURE': ambient,
             'MODULE_TEMPERATURE': module,
             'IRRADIATION': irradiation}
        new_df = pd.DataFrame(data = d, index = ambient.index)
        return(new_df)

sp.call('clear', shell = True)

#Import datasets for Plant 1
eod_df = PrepGenerationData('data/Plant_1_Generation_Data.csv').prep_data()

#ax = sns.lineplot(x = eod_df.index, y = 'DAILY_YIELD',data = eod_df,
#                  hue = 'SOURCE_KEY')
#ax.get_figure().savefig('figures/Daily_Yield_for_Inverters_Plant_1.png')
#ax.get_figure().clf()

#ax = sns.pairplot(data = eod_df)
#ax.get_figure().savefig('figures/Correlation_Matrix.png')
#ax.get_figure().clf()

#Run Ridge Regression on each inverter's data.
ridge_score = []
reg = linear_model.RidgeCV(normalize = True)
for inverter in eod_df['SOURCE_KEY'].unique():
    eod_df_test = eod_df[eod_df['SOURCE_KEY'] == inverter]
    X = eod_df_test[['AC_POWER', 'DC_POWER']]
    y = eod_df_test['DAILY_YIELD']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size = 0.1, shuffle = False,
                                                      random_state = 12)
    reg.fit(X_train, y_train)
    ridge_score.append(reg.score(X_val, y_val))
    
ridge_score = np.asarray(ridge_score)

#Run Lasso Regression on the same data
lasso_score = []
reg = linear_model.LassoCV(normalize = True, max_iter = 10000,
                               random_state = 12)
for inverter in eod_df['SOURCE_KEY'].unique():
    eod_df_test = eod_df[eod_df['SOURCE_KEY'] == inverter]
    X = eod_df_test[['AC_POWER', 'DC_POWER']]
    y = eod_df_test['DAILY_YIELD']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size = 0.1, shuffle = False,
                                                      random_state = 12)
    reg.fit(X_train, y_train)
    lasso_score.append(reg.score(X_val, y_val))
    
lasso_score = np.asarray(lasso_score)

#Add the data we have from the weather sensor and run Lasso Regression again
weather_df = PrepWeatherData('data/Plant_1_Weather_Sensor_Data.csv').prep_data()
lasso_with_weather_score = []
for inverter in eod_df['SOURCE_KEY'].unique():
    eod_df_test = eod_df[eod_df['SOURCE_KEY'] == inverter].join(weather_df)
    X = eod_df_test.drop(columns = ['SOURCE_KEY', 'DAILY_YIELD'])
    y = eod_df_test['DAILY_YIELD']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size = 0.1, shuffle = False,
                                                      random_state = 12)
    reg.fit(X_train,y_train)
    lasso_with_weather_score.append(reg.score(X_val, y_val))

lasso_with_weather_score = np.asarray(lasso_with_weather_score)

#Create a chart to display what's wrong with inverters 0 and 11
eod_df['mark'] = np.zeros(shape = len(eod_df))
eod_df['mark'] = eod_df['mark'].mask(eod_df['SOURCE_KEY'] == 0, 1).mask(eod_df['SOURCE_KEY'] == 11, 1)    

#ax2 = sns.lineplot(x = eod_df.index, y = 'DAILY_YIELD', data = eod_df,
#                   hue = eod_df.mark)
#ax2.get_figure().savefig('figures/Comparing_Underperforming_Inverters.png')
#ax2.get_figure().clf()

#Let's test it out on Plant 2
eod_df_2 = PrepGenerationData('data/Plant_2_Generation_Data.csv').prep_data()
weather_df_2 = PrepWeatherData('data/Plant_2_Weather_Sensor_Data.csv').prep_data()
df_2_score = []
for inverter in eod_df_2['SOURCE_KEY'].unique():
    eod_df_test = eod_df_2[eod_df_2['SOURCE_KEY'] == inverter].join(weather_df_2)
    X = preprocessing.scale(eod_df_test.drop(columns = ['SOURCE_KEY', 'DAILY_YIELD']))
    y = preprocessing.scale(eod_df_test['DAILY_YIELD'])
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.1, shuffle = False,
                                                        random_state = 12)
    reg.fit(X_train, y_train)
    df_2_score.append(reg.score(X_test,y_test))
df_2_score = np.asarray(df_2_score)
del(X, X_train, X_val, X_test, eod_df_test, inverter, y, y_train, y_val, y_test)

#ax3 = sns.lineplot(x = eod_df_2.index, y = 'DAILY_YIELD', data = eod_df_2,
#                  hue = eod_df.SOURCE_KEY, palette = 'husl')
#ax3.get_figure().savefig('figures/Daily_Yield_for_Inverters_Plant_2.png')
#ax3.get_figure().clf()