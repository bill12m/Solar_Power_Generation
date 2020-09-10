import subprocess as sp
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split

sp.call('clear', shell = True)

#Import datasets for Plant 1
eod_df = pd.read_csv('data/Plant_1_End_of_Day.csv').set_index('DATE_TIME')
ax = sns.lineplot(x = eod_df.index, y = 'DAILY_YIELD',data = eod_df,
                  hue = 'SOURCE_KEY')
correlation = sns.pairplot(data = eod_df)

#Run Ridge Regression on each inverter's data.
first_score = []
for inverter in eod_df['SOURCE_KEY'].unique():
    eod_df_test = eod_df[eod_df['SOURCE_KEY'] == inverter]
    X = eod_df_test[['AC_POWER', 'DC_POWER']]
    y = eod_df_test['DAILY_YIELD']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size = 0.1, shuffle = False,
                                                      random_state = 12)
    reg = linear_model.Ridge(alpha = 0.5)
    reg.fit(X_train, y_train)
    first_score.append(reg.score(X_val, y_val))
    
first_score = np.asarray(first_score)
# The score array contains the R^2 coefficient for each of the regression models
#Right now, 11%-28% of each model's variance is unexplained by just current
#variables
weather_df = pd.read_csv('data/Plant_1_Average_Weather_Stats.csv').set_index('DATE_TIME')

second_score = []
for inverter in eod_df['SOURCE_KEY'].unique():
    eod_df_test = eod_df[eod_df['SOURCE_KEY'] == inverter].join(weather_df)
    X = eod_df_test.drop(columns = ['SOURCE_KEY', 'DAILY_YIELD'])
    y = eod_df_test['DAILY_YIELD']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size = 0.1, shuffle = False,
                                                      random_state = 12)
    reg = linear_model.Ridge(alpha = 0.5)
    reg.fit(X_train,y_train)
    second_score.append(reg.score(X_val, y_val))

second_score = np.asarray(second_score)
print('Max: ', np.max(second_score), '\nMin: ', np.min(second_score))
#Absolutely no improvement adding the other variables.