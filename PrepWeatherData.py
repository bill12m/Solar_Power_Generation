import pandas as pd
import subprocess as sp

sp.call('clear', shell = True)

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