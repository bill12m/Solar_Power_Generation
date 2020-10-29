import subprocess as sp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from datetime import time

sp.call('clear', shell = True)

class PrepGenerationData:
    def __init__(self,filename):
        self.filename = filename
    
    def prep_data(self):        
        df = pd.read_csv(self.filename).drop(columns = ['PLANT_ID','TOTAL_YIELD'],axis = 1)
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], dayfirst = True)
        df['date'] = pd.to_datetime(df['DATE_TIME'].dt.date)
        df['time'] = df['DATE_TIME'].dt.time
        df = df.sort_values(['SOURCE_KEY', 'DATE_TIME']).set_index('DATE_TIME')
        
        #Convert SOURCE_KEY from object to integer we can use
        new_sourcekey_num = list(np.arange(0,22))
        old_source_key = list(df['SOURCE_KEY'].unique())
        for n in range(len(old_source_key)):
            df = df.replace(old_source_key[n],new_sourcekey_num[n])
        del(old_source_key,new_sourcekey_num,n)
        return(df)
def multi_plot(data= None, row = None, col = None, title='Daily Yield', inverter = None):
    cols = data.columns # take all column
    gp = plt.figure(figsize=(20,20)) 
    
    gp.subplots_adjust(wspace=0.2, hspace=0.8)
    for i in range(1, len(cols)+1):
        ax = gp.add_subplot(row,col, i)
        data[cols[i-1]].plot(ax=ax, style = 'k.')
        ax.set_title('{} {}'.format(title, cols[i-1]))
    name = str(inverter)
    filename = 'figures/EDA_Bad_Sensors/Date_vs_Time_per_Inverter/Plant_1/Daily_Yield_Inverter_%s.png' % name
    gp.savefig(filename)
    

#Import and clean generation data
df = PrepGenerationData(('data/Plant_1_Generation_Data.csv')).prep_data()
df_avg = df.groupby(['SOURCE_KEY','date'])['DC_POWER','AC_POWER'].agg('mean')

#Plot the daily yield per day for each inverter
for inverter in df['SOURCE_KEY'].unique():
    df_inverter = df[df['SOURCE_KEY'] == inverter]
    df_daily_yield = df_inverter.pivot('time', 'date', 'DAILY_YIELD').fillna(method = 'bfill')
    multi_plot(df_daily_yield, row = 9, col = 4,
               inverter = inverter)
    #Average results for each inverter
    df_avg_inverter = df_avg.xs(inverter)
    g = sns.histplot(data = df_avg_inverter, x = 'DC_POWER', kde = True)
    name = str(inverter)
    filename = 'figures/EDA_Bad_Sensors/Avg_DC_Power_per_Inverter/Plant_1/DC_Inverter_%s.png' % name
    g.get_figure().savefig(filename)
    


#eod_describe = []
#for inverter in eod_df['SOURCE_KEY'].unique():
#    df_inverter = eod_df[eod_df['SOURCE_KEY'] == inverter]
#    eod_describe.append(df_inverter['DAILY_YIELD'].describe())
#eod_describe_scaled = np.asarray(eod_describe)
#min_max_scaler = preprocessing.MinMaxScaler()
#eod_describe_scaled = min_max_scaler.fit_transform(eod_describe_scaled)
#eod_stats = pd.DataFrame(data = eod_describe_scaled, 
#                         index = eod_df['SOURCE_KEY'].unique(),
#                         columns = eod_describe[0].index).drop('count', axis = 1)
    