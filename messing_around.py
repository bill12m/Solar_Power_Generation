import subprocess as sp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, AffinityPropagation

sp.call('clear', shell = True)
sns.set_theme()

class PrepWeatherData:
    def __init__(self,filename):
        self.filename = filename
    
    def prep_data(self):
        df_weather = pd.read_csv(self.filename).drop(columns = 'PLANT_ID', axis = 1)
        df_weather['DATE_TIME'] = pd.to_datetime(df_weather['DATE_TIME'], dayfirst = True)
        #Setting the frequency of a DateTimeIndex creates NaN values
        #Interpolate the NaNs from the existing data to fill them.
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
def multi_plot(data= None, row = None, col = None, title='Daily Yield', inverter = None, plant = None):
    cols = data.columns # take all column
    gp = plt.figure(figsize=(20,20)) 
    
    gp.subplots_adjust(wspace=0.2, hspace=0.8)
    for i in range(1, len(cols)+1):
        ax = gp.add_subplot(row,col, i)
        data[cols[i-1]].plot(ax=ax, style = 'k.')
        ax.set_title('{} {}'.format(title, cols[i-1]))
    name = str(inverter)
    filename = 'figures/EDA_Bad_Sensors/Date_vs_Time_per_Inverter/Plant_' + plant + '/Daily_Yield_Inverter_%s.png' % name
    gp.savefig(filename)
    plt.close(gp)
    

df = pd.read_csv('data/Plant_1_Weather_Sensor_Data.csv').drop(columns = ['PLANT_ID']).append(
        pd.read_csv('data/Plant_2_Weather_Sensor_Data.csv').drop(columns = ['PLANT_ID']))

new_sourcekey_num = list(np.arange(0,df['SOURCE_KEY'].nunique()))
old_source_key = list(df['SOURCE_KEY'].unique())
for n in range(len(old_source_key)):
    df = df.replace(old_source_key[n],new_sourcekey_num[n])
del(old_source_key,new_sourcekey_num,n)
normalizer = Normalizer()
df['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION'] = normalizer.fit(
    df[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION']].values)
