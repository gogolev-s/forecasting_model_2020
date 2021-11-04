import pandas as pd
import requests
import datetime
import os

# =============================================================================
# преобразуем прогноз и загружаем реальные значения прогноза на график
# график показывает реальное значение прогноза без ручного вмешательства
# =============================================================================
def send_forecast_to_plot(df_func, products, last_trans_date, day_after_today, forecast_url, forecast_plan_url, plans_url):

    token = os.getenv('ML_API_KEY')
    model_name = os.getenv('model_name')
    account = os.getenv('account')

    ##### Forecast in pieces
    result = df_func[df_func['date'] >= (last_trans_date + datetime.timedelta(days=day_after_today))][:]
    result['value'] = result['sales']
    result['dt'] = result['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    result['prediction_type'] = 'graph_amount'
    result['model'] = model_name
    result['account_id'] = account
    result = result[['dt', 'value', 'prediction_type', 'model', 'item_id', 'place_id', 'account_id']]
    result.drop_duplicates(['place_id', 'dt', 'item_id'], inplace=True)
    data = result.to_dict('records')
    example_forecats = {'token': token, 'forecasts': data}
    r = requests.post(url = forecast_url, json = example_forecats)
    print(r.text)

    ##### Forecast in money
    result = result.merge(products[['item_id', 'place_id', 'item_discount']].drop_duplicates(['item_id', 'place_id']), on=['item_id', 'place_id'], how='left')
    result['prediction_type'] = 'graph_total'
    result['value'] = result['value'] * result['item_discount']
    result = result[['dt', 'value', 'prediction_type', 'model', 'item_id', 'place_id', 'account_id']]
    result.drop_duplicates(['place_id', 'dt', 'item_id'], inplace=True)
    result.dropna(axis=0, inplace=True)
    data = result.to_dict('records')
    example_forecats = {'token': token, 'forecasts': data}
    r = requests.post(url = forecast_url, json = example_forecats)
    print(r.text)

    ##### "Financial plan" daily forecast
    result.loc[:,'date'] = result['dt'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
    result = result[(result['date'] - last_trans_date > datetime.timedelta(days=0)) & (result['date'] - last_trans_date <= datetime.timedelta(days=day_after_today))]

    plans = pd.DataFrame.from_dict(requests.get(plans_url).json()['plans'])
    if plans.shape[0] > 0:
        plans = plans[plans['date'].apply(lambda x: int(x[5:7])) == last_trans_date.month]
        plans['id'] = plans['point'].apply(lambda x: x['id'])
        result = result[~(result.place_id.isin(plans.id.unique()))]
    result['value'] = result['value'] * 1.05
    result = result.groupby(['place_id', 'dt'])['value'].sum().reset_index()
    result['model'] = model_name
    result['account_id'] = account
    result['prediction_type'] = 0
    result = result[['dt', 'value', 'model', 'account_id', 'place_id', 'prediction_type']]
    result.dropna(inplace=True)
    data = result.to_dict('records')
    example_forecats = {'token': token, 'forecasts': data}
    r = requests.post(url = forecast_plan_url, json = example_forecats)
    print(r.text)
    print('Send kpi daily, answer code: ', r.status_code)
    return result

# =============================================================================
# преобразуем прогноз и загружаем значения в таблицы для работы сотрудников
# таблица учитывает все ручные корректировки
# =============================================================================
def send_forecast_to_table(df_func, last_trans_date, day_after_today, forecast_url, settings=pd.DataFrame(), min_value=pd.DataFrame()):
    empty_settings = settings.shape[0] == 0

    token = os.getenv('ML_API_KEY')
    model_name = os.getenv('model_name')
    account = os.getenv('account')

    result = df_func[df_func['date'] >= (last_trans_date + datetime.timedelta(days=day_after_today))][:]
    result['value'] = result['sales']
    #result['Weekday']  = result['date'].apply(lambda x: int(x.weekday()))
    if not empty_settings:
        result = result.merge(settings[['item_id', 'min', 'plan', 'digits']], on=['item_id'], how='left')
        result['digits'].fillna(0, inplace=True)
        result['min'].fillna(0, inplace=True)
        result['plan'].fillna(0, inplace=True)
        result['value'] = result.apply(lambda x: x['value'] + x['plan'], axis=1)
        result['value'] = result.apply(lambda x: max(x['value'], x['min']), axis=1)
        result['value'] = result.apply(lambda x: round(x['value'], int(x['digits'])), axis=1)
    else:
        result['value'] = result['value'].apply(lambda x: round(x, 0))

    result.drop_duplicates(['date', 'place_id', 'item_id'], inplace=True)
    result['dt'] = result['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    result['prediction_type'] = 'order_daily' #101
    result['model'] = model_name
    result['account_id'] = account
    result = result[['dt', 'value', 'prediction_type', 'model', 'item_id', 'place_id', 'account_id']]
    result.drop_duplicates(['place_id', 'dt', 'item_id'], inplace=True)
    data = result.to_dict('records')
    example_forecats = {'token': token, 'forecasts': data}

    r = requests.post(url = forecast_url, json = example_forecats)
    print(r.text)
    return result