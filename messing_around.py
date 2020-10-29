import subprocess as sp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import cluster
from datetime import time

sp.call('clear', shell = True)
sns.set_theme()

class PrepGenerationData:
    def __init__(self,filename):
        self.filename = filename
    
    def prep_data(self):        
        df = pd.read_csv(self.filename).drop(columns = ['PLANT_ID','TOTAL_YIELD'],axis = 1)
        
        #Normalize DC_POWER, AC_POWER, and DAILY_YIELD
        min_max_scaler = preprocessing.MinMaxScaler()
        df[['DC_POWER','AC_POWER','DAILY_YIELD']] = min_max_scaler.fit_transform(df[['DC_POWER','AC_POWER','DAILY_YIELD']].values)
        
        #Convert datetime to correct format, create new columns for date and time
        #Set datetime as the index
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], dayfirst = True)
        df['date'] = pd.to_datetime(df['DATE_TIME'].dt.date)
        df['time'] = df['DATE_TIME'].dt.time
        df = df.sort_values(['SOURCE_KEY', 'DATE_TIME']).set_index('DATE_TIME')
        #Extra command for cleaning up data from Plant 2
        df.iloc[df.index.indexer_between_time(time(0), time(4)), -3] = 0
        
        #Convert SOURCE_KEY from object to integer we can use
        #new_sourcekey_num = list(np.arange(0,22))
        #old_source_key = list(df['SOURCE_KEY'].unique())
        #for n in range(len(old_source_key)):
        #    df = df.replace(old_source_key[n],new_sourcekey_num[n])
        #del(old_source_key,new_sourcekey_num,n)
        return(df)
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
    

df = PrepGenerationData('data/Plant_1_Generation_Data.csv').prep_data().append(
        PrepGenerationData('data/Plant_2_Generation_Data.csv').prep_data())
new_sourcekey_num = list(np.arange(0,df['SOURCE_KEY'].nunique()))
old_source_key = list(df['SOURCE_KEY'].unique())
for n in range(len(old_source_key)):
    df = df.replace(old_source_key[n],new_sourcekey_num[n])
del(old_source_key,new_sourcekey_num,n)
        

eod_describe = df.groupby('SOURCE_KEY')['DAILY_YIELD'].describe()
kmeans = cluster.KMeans(n_clusters = 2, random_state = 12)
X = eod_describe.values
kmeans.fit(X)

clustering = cluster.AffinityPropagation(random_state = 12,
                                         max_iter = 1000).fit(X)
eod_describe['class'] = clustering.labels_
class_3 = eod_describe[eod_describe['class'] == 3]
