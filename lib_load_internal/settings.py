import pandas as pd
import requests


#выгружаем
#смотрим
#выбираем минимальное
#сохраняем для себя
# =============================================================================
# Выкачиваем данные по планограмме 
# =============================================================================
def create_settings(places, set_url, products, token_url):
    df_new = pd.DataFrame()
    for cur_id in places['place_id'].drop_duplicates():
        settings_url = set_url + cur_id + '/items_settings?' + token_url        
        json_temp = requests.get(settings_url).json()
        if  not ('settings' in json_temp):
            return df_new, pd.DataFrame(), False
        temp =  pd.DataFrame.from_dict(json_temp['settings'])
        df_new = pd.concat([df_new, temp],ignore_index=True)
        
    if df_new.shape[0]>0:
        flag = False
        df_new.rename(columns={'min_value_morning':'min', 'min_value_evening':'plan', 'week_day':'Weekday'}, inplace=True)
        df_new['Weekday'].replace({'Mo':0, 'Tu':1, 'We':2, 'Th':3, 'Fr':4, 'Sa':5, 'Su':6}, inplace=True)
        df_new.loc[:,'min'] = df_new.apply(lambda x: max(x['min'], x['plan']), axis=1)
        df_new['digits'] = 0

        mins = df_new[df_new['min'] > 0][['item_id', 'place_id', 'min', 'Weekday']]
        mins = mins.merge(products, on = ['item_id', 'place_id'], how='left')
    else:
        flag = True
        mins = pd.DataFrame()

    return df_new, mins, flag