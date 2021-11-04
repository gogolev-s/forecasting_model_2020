import requests
import datetime
import os
import pandas as pd
from pytz import timezone

# =============================================================================
# преобразуем прогноз и загружаем реальные значения прогноза на график и в таблицу
# =============================================================================

def to_table(df_func, last_trans_date, day_after_today, forecast_url, tz, empty_settings=True, settings=None):

    token = os.getenv('ML_API_KEY')
    model_name = os.getenv('model_name')
    account = os.getenv('account')

    result = df_func[df_func['date'] >= (last_trans_date + datetime.timedelta(days=day_after_today))][:]
    result['value'] = result['sales']
    result['Weekday']  = result['date'].apply(lambda x: int(x.weekday()))
    if not empty_settings:
        result = result.merge(settings[['item_id', 'place_id', 'Weekday', 'min', 'plan']], on=['item_id', 'place_id', 'Weekday'], how='left')
        result[['plan', 'min']].fillna(0, inplace=True)
        result_evening = result[result['hour']>=18].groupby(['item_id', 'place_id', 'date'])['sales'].sum().rename(columns={'value':'evening_sales'}).reset_index()
        result = result[result['hour']<=18][:]
        result = result.merge(result_evening, on=['item_id', 'place_id', 'date'], how='left')
        result['value'] = result.apply(lambda x: max(x['evening_sales'], x['plan']) if x['hour']==18 else max(x['value'], 0), axis=1)
        result['value'] = result['value'].apply(lambda x: round(x, 0))
        #for i in range(0, 14):
            #result_temp = result[result['date']==last_trans_date + datetime.timedelta(days=i + 2)]
            #rr = mins[mins['Weekday']==(last_trans_date + datetime.timedelta(days=i + 2)).weekday()][['item_id', 'place_id', 'min', 'Weekday']].merge(result_temp.drop_duplicates(['item_id', 'place_id']), on=['item_id', 'place_id', 'Weekday'], how='left', indicator=True)
            #rr = rr[rr['_merge']=='left_only'][['item_id', 'place_id', 'min_x']].rename(columns={'min_x':'value'})
            #rr['date'] = last_trans_date + datetime.timedelta(days=i + 2)
            #if rr.shape[0] > 0 : rr['Weekday']  = rr['date'].apply(lambda x: int(x.weekday()))
            #result = pd.concat([result, rr], sort=False)
    else:
        result['value'] = result['value'].apply(lambda x: round(x, 0))

    result.drop_duplicates(['date', 'place_id', 'item_id', 'hour'], inplace=True)
    result['dt'] = result.apply(lambda x: (datetime.datetime(year=x['date'].year, month=x['date'].month, day=x['date'].day, hour = int(x['hour']), tzinfo=timezone(tz)).astimezone(timezone('UTC'))).isoformat(), axis=1)
    result['prediction_type'] = 3
    result['model'] = model_name
    result['account_id'] = account
    result = result[['dt', 'value', 'prediction_type', 'model', 'item_id', 'place_id', 'account_id']]
    result.drop_duplicates(['place_id', 'dt', 'item_id'], inplace=True)
    data = result.to_dict('records')
    example_forecats = {'token': token, 'forecasts': data}

    r = requests.post(url = forecast_url, json = example_forecats)
    print(r.text)

    return result

def to_plot(df_func, products, last_trans_date, day_after_today, forecast_url, tz, empty_settings=True, settings=None):

    token = os.getenv('ML_API_KEY')
    model_name = os.getenv('model_name')
    account = os.getenv('account')

    result = df_func[df_func['date'] >= (last_trans_date + datetime.timedelta(days=day_after_today))][:]
    result['value'] = result['sales']
    result['Weekday']  = result['date'].apply(lambda x: int(x.weekday()))
    if not empty_settings:
        result = result.merge(settings[['item_id', 'place_id', 'Weekday', 'min', 'plan']], on=['item_id', 'place_id', 'Weekday'], how='left')
        result[['plan', 'min']].fillna(0, inplace=True)
        result_evening = result[result['hour']>=18].groupby(['item_id', 'place_id', 'date'])['sales'].sum().rename(columns={'value':'evening_sales'}).reset_index()
        result = result[result['hour']<=18][:]
        result = result.merge(result_evening, on=['item_id', 'place_id', 'date'], how='left')
        result['value'] = result.apply(lambda x: max(x['evening_sales'], x['plan']) if x['hour']==18 else max(x['value'], 0), axis=1)
        result['value'] = result['value'].apply(lambda x: round(x, 0))
        #for i in range(0, 14):
            #result_temp = result[result['date']==last_trans_date + datetime.timedelta(days=i + 2)]
            #rr = mins[mins['Weekday']==(last_trans_date + datetime.timedelta(days=i + 2)).weekday()][['item_id', 'place_id', 'min', 'Weekday']].merge(result_temp.drop_duplicates(['item_id', 'place_id']), on=['item_id', 'place_id', 'Weekday'], how='left', indicator=True)
            #rr = rr[rr['_merge']=='left_only'][['item_id', 'place_id', 'min_x']].rename(columns={'min_x':'value'})
            #rr['date'] = last_trans_date + datetime.timedelta(days=i + 2)
            #if rr.shape[0] > 0 : rr['Weekday']  = rr['date'].apply(lambda x: int(x.weekday()))
            #result = pd.concat([result, rr], sort=False)
    else:
        result['value'] = result['value'].apply(lambda x: round(x, 0))

    result.drop_duplicates(['date', 'place_id', 'item_id', 'hour'], inplace=True)
    result['dt'] = result.apply(lambda x: datetime.datetime(year=x['date'].year, month=x['date'].month, day=x['date'].day, hour = int(x['hour']), tzinfo=timezone(tz)).astimezone(timezone('UTC')), axis=1)
    result['dt'] = result['dt'].apply(lambda x: x.replace(minute=0).isoformat())
    result['prediction_type'] = 'graph_amount_hourly'
    result['model'] = model_name
    result['account_id'] = account
    result = result[['dt', 'value', 'prediction_type', 'model', 'item_id', 'place_id', 'account_id']]
    result.drop_duplicates(['place_id', 'dt', 'item_id'], inplace=True)
    data = result.to_dict('records')
    example_forecats = {'token': token, 'forecasts': data}
    r = requests.post(url = forecast_url, json = example_forecats)
    print(r.text)

    result['prediction_type'] = 'graph_total_hourly'
    result = result.merge(products[['item_id', 'item_discount']].drop_duplicates(['item_id']), how='left', on='item_id')
    result['item_discount'].fillna(0, inplace=True)
    result['value'] = result['value'] * result['item_discount']
    result = result[['dt', 'value', 'prediction_type', 'model', 'item_id', 'place_id', 'account_id']]
    data = result.to_dict('records')
    example_forecats = {'token': token, 'forecasts': data}
    r = requests.post(url = forecast_url, json = example_forecats)
    print(r.text)

    return result
