import pandas as pd
import datetime
import lightgbm as lgb
import requests
import numpy as np
from time import sleep
import re
from datetime import timedelta
import pickle
import os
from imblearn.over_sampling import SMOTE

from lib_preprocess import create_aggr


def create_bin_model(df_func, space_num, target, test_date):
    not_columns = ['item_id' , 'place_id' , 'sales' , 'item_category' , 'item_name' , 'category_id', 'expiry', 'expiry_sum', 'cost_price', 'city' , 'clouds_all' , 'dt', 'date','datetime' , 'weather', 'icon']
    if 'morning_sales' in df_func.columns:
        not_columns = not_columns + ['morning_sales']
    train_cols = list(set(df_func.columns) - set(not_columns))
    df_train = df_func[df_func['date'] <=  test_date].copy()
    df_train = df_train.replace(-np.inf, np.nan).replace(np.inf, np.nan).fillna(0)
    def get_temp_description(temp):
        return {temp < 1:0, 1 <= temp < 2: 1, 2 <= temp: 2}[True]
    sales = df_train[target]
    Y_train = sales.apply(lambda x: get_temp_description(x))
    Y_train = Y_train.astype('int')
    X_train = df_train.drop([target], axis=1, inplace=False)
    del df_train
    sleep(1)
    X_train = X_train[train_cols]
    
    
    ov = SMOTE(random_state=0)


    X_train = X_train.replace(-np.inf, np.nan).replace(np.inf, np.nan).fillna(0)
    X_resampled, y_resampled = ov.fit_resample(X_train, Y_train)
    lgb_tt = lgb.Dataset(X_resampled, y_resampled)
# =============================================================================
#     Классификатор
# =============================================================================
    params_bin = {'bagging_fraction': 0.94,
                 'bagging_freq': 100,
                 'boosting_type': 'gbdt',
                 'feature_fraction': 0.94,
                 'learning_rate': 0.08,
                 #'max_depth': 250,
                 'min_data_in_leaf': 225,
                 'num_leaves': 200,
                 'num_class':3,
                 'verbose': -1,
                 'objective': 'multiclass'}
    model_bin = lgb.train(params_bin, lgb_tt, verbose_eval=False, num_boost_round = 500)
    
# =============================================================================
#     Регрессия 
# =============================================================================
    
    params_reg ={'alpha':0.66,
                 'bagging_freq': 20, #220
                 'boosting_type': 'gbdt',
                 'learning_rate': 0.06, #0.12
                 'max_depth': 1000, #340
                 'min_data_in_leaf': 80, #150
                 'num_leaves': 600, #350
                 'objective': 'quantile',
                 'verbose': -1,
                 'metric': 'quantile'}
    
    X_train, Y_train = X_train[sales > 1],  sales[sales > 1]
    
    lgb_tt = lgb.Dataset(X_train[:], Y_train[:])
    model_reg = lgb.train(params_reg, lgb_tt,  verbose_eval=False, num_boost_round = 500)
    
    return model_bin, model_reg, train_cols

def transform_day_to_hour(df_func, future_df_func, df_hour_func):
    
    df_func = df_func[df_func['date']< future_df_func['date'].min()]
    df_func = pd.concat([df_func, future_df_func], sort=False)
    df_func.rename(columns={'sales':'day_sales'}, inplace=True)
    both_cols = list(set(df_func.columns) & set(df_hour_func.columns) - {'place_id', 'item_id', 'date'})
    df_hour_func = df_hour_func.merge(df_func.drop(both_cols, axis=1), on=['place_id', 'item_id', 'date'], how='left')
    df_hour_func.fillna(0, inplace=True)
    return df_hour_func

# =============================================================================
# Делаем прогноз на следующие дни
# =============================================================================
def create_forecast(df_func, model_class, model_reg, train_cols, predict_days, weeks_lag, days_lag, with_weather, test_date):
    target = 'sales'
    pref = '_h_'
    additional_key =['hour']
    # предсказанные данные
    future_df = pd.DataFrame()
    # реальные исторические данные
    lagged_df = df_func[(df_func['date'] >= test_date - datetime.timedelta(max(weeks_lag * 7, days_lag) + 2))&(df_func['date'] <= test_date)].copy()
    for d in range(predict_days):
        step_date = test_date + datetime.timedelta(d + 1)
        old_df = pd.concat([lagged_df, future_df], sort=False)
        new_df = df_func[df_func['date'] == (test_date + datetime.timedelta(days=1+d))]
        new_df = pd.concat([old_df, new_df], sort=False)
        new_df = create_aggr.add_week_statistics(df_func = new_df, weeks_lag = weeks_lag, additional_key = additional_key)
        for new_lag in range(1, max(1, d+1)):
            date_i = 'date_' + str(new_lag) + 'ago'
            new_df[date_i] = new_df['date'].apply(lambda x: x - datetime.timedelta(days=new_lag))
            new_df.drop('sales' + pref + str(new_lag) + 'ago', axis=1, inplace=True)
            columns_right    = ['date','item_id', 'place_id', 'sales'] + additional_key
            columns_left_on  = [date_i,'item_id', 'place_id'] + additional_key
            columns_right_on = ['date','item_id', 'place_id'] + additional_key
            sleep(0.1)
            new_df = new_df.merge(old_df[columns_right], left_on=columns_left_on, right_on=columns_right_on, suffixes=('', pref + str(new_lag) + 'ago'), how='left').drop_duplicates()
        # удаляем столбцы с лаговыми датами
        new_df.drop(columns = [i for i in new_df.columns if i.startswith('date') & i.endswith('ago')], axis=1, inplace=True)
        # отбираем столбцы с лаговыми значениями
        sales_columns = [i for i in new_df.columns if i.startswith('sales' + pref) & i.endswith('ago')]
        sales_week_columns  = [i for i in sales_columns if (int(re.findall(r'\d+', i)[0])<=7)]
        sales_week2_columns = [i for i in sales_columns if (int(re.findall(r'\d+', i)[0])<=14)]
        if days_lag > 21:
            new_df['w' + pref + 'avg_vtornik'] = new_df[['sales' + pref + '7ago', 'sales' + pref + '14ago', 'sales' + pref + '21ago']].mean(axis=1)
            new_df = create_aggr.add_statistics(new_df, sales_week_columns, 'w' + pref)
            new_df = create_aggr.add_statistics(new_df, sales_week2_columns, 'w2' + pref)
            new_df['w' + pref + 'ratio_1vtornik'] = new_df['sales' + pref + '1ago'] / new_df['w' + pref + 'avg_vtornik']
            new_df['w' + pref + 'ratio_2vtornik'] = new_df['sales' + pref + '2ago'] / new_df['w' + pref + 'avg_vtornik']
            new_df['w2' + pref + 'ratio_vtornik'] = new_df['sales' + pref + '7ago'] / new_df['w' + pref + 'avg_vtornik']
            new_df['w' + pref + 'ratio'] = new_df['w' + pref + 'sum'] / new_df['w2' + pref + 'avg']
            new_df['w' + pref + 'ratio_avg'] = new_df['w' + pref + 'avg'] / new_df['w2' + pref + 'avg']
        elif days_lag >= 14:
            new_df['w' + pref + 'avg_vtornik'] = new_df[['sales' + pref + '7ago', 'sales' + pref + '14ago']].mean(axis=1)
            new_df = create_aggr.add_statistics(new_df, sales_week_columns, 'w' + pref)
            new_df = create_aggr.add_statistics(new_df, sales_week2_columns, 'w2' + pref)
            new_df['w' + pref + 'ratio_1vtornik'] = new_df['sales' + pref + '1ago'] / new_df['w' + pref + 'avg_vtornik']
            new_df['w' + pref + 'ratio_2vtornik'] = new_df['sales' + pref + '2ago'] / new_df['w' + pref + 'avg_vtornik']
            new_df['w2' + pref + 'ratio_vtornik'] = new_df['sales' + pref + '7ago'] / new_df['w' + pref + 'avg_vtornik']
            new_df['w' + pref + 'ratio'] = new_df['w' + pref + 'sum'] / new_df['w2' + pref + 'avg']
            new_df['w' + pref + 'ratio_avg'] = new_df['w' + pref + 'avg'] / new_df['w2' + pref + 'avg']
        else:
            new_df['w' + pref + 'avg_vtornik'] = new_df['sales' + pref + '7ago']
        if with_weather:
            for i in range (1, 3):
                for var in ['temp', 'humidity', 'temp_min', 'temp_max', 'wind_speed', 'pressure']:
                    new_df[var + pref + str(i) +'ratio'] = new_df.apply(lambda x: x['sales' + pref + str(i) + 'ago'] if x[var + pref + str(i) + 'ago'] == 0 else x[var] / x[var + pref + str(i) + 'ago'] * x['sales' + pref + str(i) + 'ago'], axis=1)
                    
                    
# =============================================================================
#       Построение прогноза    
# =============================================================================
        new_df.fillna(0, inplace=True)
        df_test  = new_df[new_df['date'] == step_date].copy()
        res = df_test[['date', 'place_id', 'item_id'] + additional_key]
        min_hour = res.hour.min()
        max_hour = res.hour.max()

        for cur_hour in range(min_hour, max_hour + 1):
            df_test_hour = df_test[df_test['hour'] == cur_hour]
            X_test = df_test_hour[train_cols]
            # Classification      
            Y = model_class.predict(X_test)
            # Выбираем наибольшую вероятность
            Y = Y.argmax(1)
            if (Y == 2).sum() > 0 :
                Y[Y == 2] = model_reg.predict(X_test[Y == 2])
            res.loc[res['hour'] == cur_hour, 'sales_pred'] = Y
            df_test.loc[df_test['hour'] == cur_hour, 'sales'] = Y
            
            i = cur_hour + 1
            df_temp = df_test.copy()
            df_temp['key' + str(i)] = df_temp['hour'] < i
            df_temp = df_temp.groupby(['date', 'item_id', 'place_id', 'key' + str(i)])['sales'].sum().reset_index()
            df_temp['hour'] = i
            df_temp = df_temp[df_temp['key' + str(i)]].rename(columns={'sales':'sales_before'})
            
            df_test = df_test.merge(df_temp, how='left', on=['date', 'item_id', 'place_id', 'hour'])
            df_test['sales_before'] = df_test[['sales_before_x', 'sales_before_y']].apply(lambda x: x['sales_before_y'] if x['sales_before_y'] >= 0 else x['sales_before_x'], axis=1)
            df_test.drop(['sales_before_x', 'sales_before_y'], axis=1, inplace=True)
                   
        
        new_df = new_df.merge(res, on=['date', 'place_id', 'item_id'] + additional_key, how='left')
        new_df['sales'] = new_df.apply(lambda x: x['sales_pred'] if x['sales']==0 else x['sales'], axis=1)
        new_df.drop(['sales_pred'], axis=1, inplace=True)
        future_df = pd.concat([future_df, new_df.loc[new_df['date'] == step_date]], sort=False)
        sleep(1)

    return future_df