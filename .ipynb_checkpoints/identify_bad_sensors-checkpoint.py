import subprocess as sp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, AffinityPropagation
from datetime import time

sp.call('clear', shell = True)
sns.set_theme()

class PrepGenerationData:
    def __init__(self,filename):
        self.filename = filename
    
    def prep_data(self):        
        df = pd.read_csv(self.filename).drop(columns = ['PLANT_ID','TOTAL_YIELD'],axis = 1)
        
        #Normalize DC_POWER, AC_POWER, and DAILY_YIELD
        normalizer = Normalizer()
        df[['DC_POWER','AC_POWER','DAILY_YIELD']] = normalizer.fit_transform(df[['DC_POWER','AC_POWER','DAILY_YIELD']].values)
        
        #Convert datetime to correct format, create new columns for date and time
        #Set datetime as the index
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], dayfirst = True)
        df['date'] = pd.to_datetime(df['DATE_TIME'].dt.date)
        df['time'] = df['DATE_TIME'].dt.time
        df = df.sort_values(['SOURCE_KEY', 'DATE_TIME']).set_index('DATE_TIME')
        #Extra command for cleaning up data from Plant 2
        df.iloc[df.index.indexer_between_time(time(0), time(4)), -3] = 0
        
        #Convert SOURCE_KEY from object to integer we can use
        new_sourcekey_num = list(np.arange(0,22))
        old_source_key = list(df['SOURCE_KEY'].unique())
        for n in range(len(old_source_key)):
            df = df.replace(old_source_key[n],new_sourcekey_num[n])
        del(old_source_key,new_sourcekey_num,n)
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
    

plants = [str(1), str(2)]
for plant in plants:
    #Import and clean generation data
    df = PrepGenerationData(('data/Plant_' + plant + '_Generation_Data.csv')).prep_data()
    
    #Plot the average DC production per inverter for each plant. AC production
    #should be comparable
    df_avg = df.groupby(['SOURCE_KEY','date'])[['DC_POWER','AC_POWER']].agg('mean')
    fig, ax = plt.subplots()
    ax = sns.FacetGrid(data = df_avg.reset_index(), col = 'SOURCE_KEY', col_wrap = 4)
    ax.map(sns.histplot, 'DC_POWER', kde = True)
    
    #Save figure
    filename = 'figures/EDA_Bad_Sensors/Avg_DC_Power_per_Inverter/Plant_'+ plant + '.png'
    ax.savefig(filename)
    plt.close(fig)
    
    #Plot the daily yield per day for each inverter
    for inverter in df['SOURCE_KEY'].unique():
        df_inverter = df[df['SOURCE_KEY'] == inverter]
        df_daily_yield = df_inverter.pivot('time', 'date', 'DAILY_YIELD').fillna(method = 'bfill', axis = 1)
        multi_plot(df_daily_yield, row = 9, col = 4, inverter = inverter, plant = plant)
        
df = pd.read_csv('data/Plant_1_Generation_Data.csv').append(
        pd.read_csv('data/Plant_2_Generation_Data.csv'))
new_sourcekey_num = list(np.arange(0,df['SOURCE_KEY'].nunique()))
old_source_key = list(df['SOURCE_KEY'].unique())
for n in range(len(old_source_key)):
    df = df.replace(old_source_key[n],new_sourcekey_num[n])
del(old_source_key,new_sourcekey_num,n)
normalizer = Normalizer()
df['DAILY_YIELD'] = normalizer.fit_transform(df[['DAILY_YIELD']].values)

        

eod_describe = df.groupby('SOURCE_KEY')['DAILY_YIELD'].describe()
#Tried PCA and didn't get different results

#Run kmeans for 4 classes. Chose 4 classes because affinity propogation determined
#4 classes and it described the data really well.
kmeans = KMeans(n_clusters = 4, random_state = 12)
kmeans.fit(eod_describe.values)
eod_describe['class_kmeans'] = kmeans.labels_

#Run affinity propogation, when choosing 4 classes, kmeans produces the same
#result.
#clustering = AffinityPropagation(random_state = 12,
#                                         max_iter = 1000).fit(eod_describe.values)
#eod_describe['class_affinity'] = clustering.labels_
