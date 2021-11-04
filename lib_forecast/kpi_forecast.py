import pandas as pd
import datetime
import lightgbm as lgb
import requests

import os



# =============================================================================
# Содаем прогнозы по kpi
# =============================================================================
def predict_analytics (forecast_plan_url, plans_url, bill_count, avg_bill, last_date):

    token = os.getenv('ML_API_KEY')
    account = os.getenv('account')
    model_name = os.getenv('model_name')
    plans = pd.DataFrame.from_dict(requests.get(plans_url).json()['plans'])
    if plans.shape[0] > 0:
        plans = plans[plans['date'].apply(lambda x: int(x[5:7])) == last_date.month]
        plans['id'] = plans['point'].apply(lambda x: x['id'])
        plans = plans[plans['sales'] + plans['receipt'] + plans['average_receipt'] > 0]
    def last_day_of_month(any_day):
        next_month = any_day.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
        return next_month - datetime.timedelta(days=next_month.day)
    for ind, df in enumerate([bill_count, avg_bill]):
        #ind = 0
        #df = bill_count
        df = df[df['date'] < last_date]
        first_mon_date = last_date.replace(day=1)
        last_date_next = last_day_of_month(first_mon_date)
        period = (last_date_next - last_date).days + 1
        new = df[df['date'] > df['date'].max() - datetime.timedelta(days=period)]
        new.loc[:,'date'] = new['date'] + datetime.timedelta(days=period)
        df = pd.concat([df, new], sort=False)
        df['weekday']= df['date'].map(lambda x: x.weekday())
        df = df.set_index('date')
        for i in range(14):
            df['value_' + str(i + 1) + 'ago'] = df.groupby(['place_id'])['value'].shift(i + 1)
        df.fillna(df['value'].mean(axis=0), inplace=True)
        df['ma'] = df[df.columns[df.columns.str.startswith('value_')]].mean(axis=1)
        df.reset_index(inplace=True)
        df = pd.merge(df, pd.get_dummies(df[['place_id', 'weekday']], columns=['place_id', 'weekday']), how='left', on=df.index, right_index=False).drop('key_0', axis=1)
        train_data = df[df['date'] < last_date]
        test_data = df[df['date'] >= (last_date)] #+ datetime.timedelta(days = 7 - last_date.weekday()))]

        Y_train = train_data['value']
        X_train = train_data.drop(['value', 'place_id', 'weekday', 'date'], axis=1, inplace=False)
        X_test = test_data.drop(['value', 'place_id', 'weekday', 'date'], axis=1, inplace=False)

        lgb_tt = lgb.Dataset(X_train[:], Y_train[:])
        model = lgb.train({'max_depth': 3, 'alpha': 0.65, 'objective':'quantile', 'bagging_freq': 40, 'learning_rate': 0.15, 'min_data_in_leaf': 25, 'num_leaves': 90}, lgb_tt, num_boost_round = 50)
        Y_test = pd.DataFrame(model.predict(X_test))
        df_new = pd.concat([test_data[['date', 'place_id']].reset_index(drop=True), Y_test[0]], axis=1).rename(columns={0:'value'})
        if ind == 1:
            df_new.loc[:,'value'] = 1.05 * df_new['value']
            avg_bill_new = df_new.copy()
        else:
            bill_count_new = df_new.copy().rename(columns={'value':'bill_count'})
        df_new = df_new[(first_mon_date <= df_new['date'])&(df_new['date']<= last_date_next)]
        df_new['dt'] = df_new['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df_new['model'] = model_name
        df_new['account_id'] = account
        df_new['prediction_type'] = ind + 1
        df_new = df_new[['dt', 'value', 'model', 'account_id', 'place_id', 'prediction_type']]
        data = df_new.to_dict('records')
        example_forecats = {'token': token, 'forecasts': data}
        r = requests.post(url = forecast_plan_url, json = example_forecats)
        print(r.text)
#df_new = df_new[df_new['dt'] == '2020-02-08'][['place_id', 'value']]
    avg_bill_new = avg_bill_new.merge(bill_count_new, on = ['date', 'place_id'], how='left')
    avg_bill_new.loc[:,'value'] = avg_bill_new['value'] * avg_bill_new['bill_count']
    df_new = avg_bill_new.copy().drop('bill_count', axis=1)

    # считаем, что получим по нашему прогнозу до конца месяца
    #total_predict = df_new[(first_mon_date <= df_new['date'])&(df_new['date']<= last_date_next)]
    #days_left = (total_predict['date'].max() - total_predict['date'].min()).days + 1
    #days_in_month = last_day_of_month(first_mon_date).day
    #total_predict = total_predict.groupby('place_id')['value'].sum().reset_index().rename(columns={'value':'pred'})
    #df_new = df_new.merge(total_predict, on='place_id', how='left')

    #df_new = df_new.merge(plans[['id', 'sales']], left_on='place_id', right_on='id', how='left')
    #df_new['sales'].fillna(df_new['pred'] * days_in_month / days_left, inplace=True)
    #df_new.loc[df_new['sales']==0, 'sales'] = df_new.loc[df_new['sales']==0, 'pred'] * days_in_month / days_left
    #df_new['sales'] = df_new['sales'] / days_in_month * days_left
    #df_new.drop('id', axis=1, inplace=True)

    #df_new['value'] = df_new['value'] * df_new['sales'] / df_new['pred']
    df_new = df_new[(first_mon_date <= df_new['date'])&(df_new['date']<= last_date_next)]
    df_new['dt'] = df_new['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df_new['model'] = model_name
    df_new['account_id'] = account
    df_new['prediction_type'] = 0
    df_new = df_new[['dt', 'value', 'model', 'account_id', 'place_id', 'prediction_type']]
    df_new.dropna(inplace=True)
    data = df_new.to_dict('records')
    example_forecats = {'token': token, 'forecasts': data}
    r = requests.post(url = forecast_plan_url, json = example_forecats)
    print(r.text)

    return r
