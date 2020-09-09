import subprocess as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

sp.call('clear', shell = True)

#Import datasets for Plant 1
eod_df = pd.read_csv('data/Plant_1_End_of_Day.csv').set_index('DATE_TIME')
#ax = sns.lineplot(x = eod_df.index, y = 'DAILY_YIELD',data = eod_df,
#                  hue = 'SOURCE_KEY')

#Run Ridge Regression on each inverter's data.
score = []
for inverter in eod_df['SOURCE_KEY'].unique():
    eod_df_test = eod_df[eod_df['SOURCE_KEY'] == inverter]
    X = eod_df_test[['DC_POWER', 'AC_POWER']]
    y = eod_df_test['DAILY_YIELD']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size = 0.1, shuffle = False,
                                                      random_state = 12)
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    score.append(reg.score(X_val, y_val))