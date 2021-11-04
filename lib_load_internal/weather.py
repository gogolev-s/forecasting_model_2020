import requests
import datetime
import os
import pandas as pd
import numpy as np
from pytz import timezone
from time import sleep



def load_weather(city_id, tz, abs_path, base_url, token_url, last_trans_date, hour_forecast=False, long_forecast=True):
    ##### Create directory if it does not exist
    if not os.path.exists(os.path.join(abs_path, "data", "weather")):
        os.mkdir(os.path.join(abs_path, "data", "weather"))

    file_path = os.path.join(abs_path, "data", "weather", str(city_id) + '.csv')
    first_date_load = last_trans_date - datetime.timedelta(days=720)
    weather = pd.DataFrame()

    if os.path.exists(file_path):
        weather = pd.read_csv(file_path, sep=',', decimal='.', parse_dates=[0], infer_datetime_format=True)
        weather.loc[:,'date'] = weather['date'].apply(lambda x: x.date())
        first_date_load = weather.date.max()
        # Reload the last day
        weather = weather[weather['date'] < first_date_load]
        weather = weather[['date', 'temp_avg', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'precipitation']]

    #### Load historical and current weather if necessary
    weather_url = base_url + 'cities/' + city_id + '/weathers?'+ token_url + '&categories[]=history&categories[]=current&from=' + str(first_date_load)
    weather_h = pd.DataFrame.from_dict(requests.get(weather_url).json()['weathers'])
    if weather_h.shape[0] > 0:
        weather_h['dt'] = weather_h['dt'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone('UTC')).astimezone(timezone(tz)))
        weather_h['date'] = weather_h['dt'].apply(lambda x: x.date())
        weather_h['precipitation'] = 1 * ((weather_h['weather'] == 'Rain') | (weather_h['weather'] == 'Snow') | (weather_h['weather'] == 'Thunderstorm'))
        weather_h = weather_h.groupby(['date']).agg(temp_avg=('temp', 'mean'),
                                                    temp_min=('temp', 'min'),
                                                    temp_max=('temp', 'max'),
                                                    pressure=('pressure', 'mean'),
                                                    humidity=('humidity', 'mean'),
                                                    wind_speed=('wind_speed', 'mean'),
                                                    precipitation=('precipitation', 'sum')).reset_index()
        weather = pd.concat([weather, weather_h], sort=False)

        weather.to_csv(file_path, sep=',', decimal='.', index=False)

    #### Load short-term forecasts
    last_load = weather.date.max() + datetime.timedelta(days=1)
    weather_url = base_url + 'cities/' + city_id + '/weathers?'+ token_url + '&categories[]=forecast&from=' + str(last_load)
    weather_f = pd.DataFrame.from_dict(requests.get(weather_url).json()['weathers'])
    if weather_f.shape[0] > 0:
        weather_f['dt'] = weather_f['dt'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone('UTC')).astimezone(timezone(tz)))
        weather_f['date'] = weather_f['dt'].apply(lambda x: x.date())
        weather_f['precipitation'] = 3 * ((weather_f['weather'] == 'Rain') | (weather_f['weather'] == 'Snow') | (weather_f['weather'] == 'Thunderstorm'))
        weather_f = weather_f.groupby(['date']).agg(temp_avg=('temp', 'mean'),
                                                    temp_min=('temp', 'min'),
                                                    temp_max=('temp', 'max'),
                                                    pressure=('pressure', 'mean'),
                                                    humidity=('humidity', 'mean'),
                                                    wind_speed=('wind_speed', 'mean'),
                                                    precipitation=('precipitation', 'sum')).reset_index()

        weather = pd.concat([weather, weather_f], sort=False)

    #### Load long-term forecasts
    if long_forecast:
        last_load = weather.date.max() + datetime.timedelta(days=1)
        weather_url = base_url + 'cities/' + city_id + '/weathers?'+ token_url + '&categories[]=forecast_long&from=' + str(last_load)
        weather_ff = pd.DataFrame.from_dict(requests.get(weather_url).json()['weathers'])
        if weather_ff.shape[0] > 0:
            weather_ff['dt'] = weather_ff['dt'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone('UTC')).astimezone(timezone(tz)))
            weather_ff['date'] = weather_ff['dt'].apply(lambda x: x.date())
            weather_ff['precipitation'] = 12 * ((weather_ff['weather'] == 'Rain') | (weather_ff['weather'] == 'Snow') | (weather_ff['weather'] == 'Thunderstorm'))
            weather_ff['temp'] = weather_ff[['temp_morn', 'temp_day', 'temp_eve', 'temp_night']].mean(axis=1)
            weather_ff = weather_ff.groupby(['date']).agg(temp_avg=('temp', 'mean'),
                                                        temp_min=('temp', 'min'),
                                                        temp_max=('temp', 'max'),
                                                        pressure=('pressure', 'mean'),
                                                        humidity=('humidity', 'mean'),
                                                        wind_speed=('wind_speed', 'mean'),
                                                        precipitation=('precipitation', 'sum')).reset_index()

            weather = pd.concat([weather, weather_ff], sort=False)
    if weather.shape[0] > 0:
        weather.reset_index(drop=True, inplace=True)
        weather.drop_duplicates('date', inplace=True)

    return weather


#погода
# =============================================================================
# Выгружаем погоду и прицепляем ее к df,
# =============================================================================
def add_weather(df_func, places, set_url, token_url, hour_forecast=False):
    cur_id = places['place_id'].iloc[0]
    tz = places['tz'].iloc[0]
    weather_url = set_url + cur_id + '/weathers?'+ token_url
    weather = pd.DataFrame.from_dict(requests.get(weather_url).json()['weathers'])
    sleep(1)
    weather['datetime'] = weather['dt'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').astimezone(timezone(tz)))
    weather['date'] = weather['datetime'].apply(lambda x: x.date())
    weather['hour'] = weather['datetime'].apply(lambda x: x.hour)
    # убираем левые строки
    weather = weather[weather['temp']>-100000]
    weather.drop_duplicates(['date', 'hour'], inplace=True)
    sleep(1)
    # ветка для прогноза по часам
    if hour_forecast:
        # прогнозы строятся каждые 3 часа, нужно прицепить их к каждому часу
        need_hour = weather.groupby(['date'])['hour'].count().reset_index()
        need_hour = need_hour[need_hour['hour']<10]['date'].values.tolist()
        weather['key'] = weather.apply(lambda x: 1 if x['date'] in need_hour else 0, axis=1)
        temp_hours = pd.DataFrame()
        for i in range(8):
            temp_hours = pd.concat([temp_hours, pd.DataFrame({'hour':[3 * i] * 3, 'h':[3 * i, 3 * i + 1, 3 * i + 2]})])
        temp_hours['key'] = 1
        weather = weather.merge(temp_hours, on=['key', 'hour'], how='left')[['date', 'h', 'hour']].merge(weather, on =['date', 'hour'], how='left').drop(['hour', 'key'], axis=1).rename(columns={'h':'hour'})
        weather = weather[['date', 'city', 'hour', 'temp', 'pressure', 'humidity',
       'temp_min', 'temp_max', 'wind_speed', 'wind_deg', 'weather', 'clouds_all']]
        df_func = df_func.merge(weather, on=['city', 'date','hour'], how='left')
        return df_func
    else:
        weather.loc[:,'temp'] = weather.apply(lambda x: x['temp'] if np.isnan(x['temp_day']) else x['temp_day'], axis=1)
        weather = weather.query('hour == 12')
        weather.drop(['hour', 'datetime'], axis=1, inplace=True)
        sleep(1)
        weather.drop_duplicates('date', inplace=True)
        sleep(1)
        df_func = df_func.merge(weather, on=['city', 'date'], how='left')
        return df_func