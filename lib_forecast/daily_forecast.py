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

from hyperopt import fmin, hp, tpe, STATUS_OK

from lib_preprocess import create_aggr



#то что продовалось неделю назад прибавляем на следующий день
# =============================================================================
# Добавляем строки на predict_days дней вперед для прогноза с учетом минимального заказа, если он есть
# =============================================================================
def add_features_for_predict(df_func, min_value, predict_days, last_date, use_settings = False, empty_settings=True):
    for i in range(predict_days):
        df_temp = df_func[df_func['date'] == last_date - datetime.timedelta(days= 6 - i)].copy()

        if df_temp.shape[0] == 0:
            df_temp = df_func[df_func['date'] == last_date - datetime.timedelta(days= 7 + 6 - i)].copy()
            #Добавляем  7 дней, которые мы вычли только что
            df_temp.loc[:,'date'] = df_temp['date'].apply(lambda x: x + datetime.timedelta(days=7))

        df_temp.loc[:,'date'] = df_temp['date'].apply(lambda x: x + datetime.timedelta(days=7))

        if (use_settings) & (~ empty_settings):
            used = min_value[min_value['Weekday']==df_temp['Weekday'].max()].merge(df_temp[['item_id', 'place_id']].drop_duplicates(), on=['item_id', 'place_id'], how='left', indicator=True)
            if df_temp.shape[0]>0 : used['date'] = df_temp['date'].drop_duplicates().values[0]
            used = used[used['_merge']=='left_only']
            used.drop(['_merge', 'min'], axis=1, inplace=True)
            df_temp = pd.concat([df_temp, used], sort=False)
        df_temp.loc[:,'sales'] = np.nan
        if 'morning_sales' in df_func.columns: df_temp.loc[:,'morning_sales'] = np.nan
        df_func = df_func.append(df_temp)
    if 'hour' in  df_func.columns:
        df_func.drop_duplicates(['date','item_id', 'place_id', 'hour'], inplace=True)
    else:
        df_func.drop_duplicates(['date','item_id', 'place_id'], inplace=True)

    return df_func.drop_duplicates()


# =============================================================================
# Обучаем модель для разных target и пространств параметров
# =============================================================================
def create_model(df_func, space_num, target, last_date):
    
    not_columns = ['item_id' , 'place_id' , 'sales' , 'item_category' , 'item_name' , 'category_id', 'expiry' , 'expiry_sum', 'cost_price', 'city' , 'clouds_all' , 'dt', 'date','datetime' , 'weather', 'icon']
    if 'morning_sales' in df_func.columns: not_columns = not_columns + ['morning_sales']
    train_cols = list(set(df_func.columns) - set(not_columns))
    df_train = df_func[df_func['date'] <= last_date].copy()
    Y_train = df_train[target]
    X_train = df_train.drop([target], axis=1, inplace=False)
    del df_train
    sleep(1)
    X_train = X_train[train_cols]
    if space_num < 1: #50 days
        params = {'alpha':0.51,
                 'bagging_fraction': 0.88,
                 'bagging_freq': 650, #220
                 'boosting_type': 'gbdt',
                 'feature_fraction': 0.6, #0.55
                 'lambda_l1': 0.6, #0.8
                 'lambda_l2': 0.4, #0.35
                 'learning_rate': 0.04, #0.12
                 'max_depth': 900, #340
                 'min_data_in_leaf': 120, #150
                 'num_leaves': 300, #350
                 'objective': 'quantile',
                 'verbose': -1,
                 'metric': 'quantile'}

    for i in params:
        if type(params[i]) != str and type(params[i]) != set:
            if int(params[i]) == params[i]:
                params[i] = int(params[i])

    lgb_tt = lgb.Dataset(X_train[:], Y_train[:])
    model = lgb.train(params=params, train_set=lgb_tt, num_boost_round=500, verbose_eval=False)

    return model, train_cols


def create_model2(df_func, abs_path, last_date, train_model_today, need_for_holidays):
    
    if need_for_holidays:
        model_path = os.path.join(abs_path, "data", "models", os.getenv('account') + '_holiday.txt')
        columns_path = os.path.join(abs_path, "data", "models", os.getenv('account') + '_holiday_id.txt')
    else:
        model_path = os.path.join(abs_path, "data", "models", os.getenv('account') + '.txt')
        columns_path = os.path.join(abs_path, "data", "models", os.getenv('account') + '_id.txt')
        
    not_columns = ['item_id' , 'place_id' , 'sales' , 'item_category' , 'item_name' , 'category_id', 'expiry' , 'expiry_sum', 'cost_price', 'city' , 'clouds_all' , 'dt', 'date','datetime' , 'weather', 'icon']
    
    if 'morning_sales' in df_func.columns: not_columns = not_columns + ['morning_sales']
    if train_model_today:
        train_cols = list(set(df_func.columns) - set(not_columns))
        
        df_train = df_func[df_func['date'] <= last_date].copy()
        
        Y_train = df_train['sales']
        X_train = df_train.drop(['sales'], axis=1, inplace=False)
        X_train = X_train[train_cols]
        
        lgb_tt = lgb.Dataset(X_train, Y_train)
        
        del df_train
        
        def objective(params):
            params = {
                'min_data_in_leaf':int(params['min_data_in_leaf']),
                'max_depth': int(params['max_depth']),
                'num_leaves': int(params['num_leaves']),  
                'bagging_freq': int(params['bagging_freq']),
                'learning_rate': '{:.3f}'.format(params['learning_rate']),          
                'boosting_type': 'gbdt',
                'objective':'quantile',
                'alpha':0.55 + 0.02 * need_for_holidays,
                'verbose': -1}
        
            cv_results = lgb.cv(
                    params,
                    lgb_tt,
                    num_boost_round=500,
                    nfold=5,
                    metrics={'mae'},
                    early_stopping_rounds=2,
                    stratified=False)
        
            #n_boost_best = len(cv_results['l1-mean'])
            mae = np.min(cv_results['l1-mean'])
        
           # print("mae {:.3f} params {} n_boost {}".format(mae, params,n_boost_best))
        
            return {'loss':mae, 'status': STATUS_OK}
    
        space = {
            'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 500, 20),
            'max_depth': hp.quniform('max_depth', 50, 900, 50),
            'num_leaves': hp.quniform('num_leaves', 50, 850, 25),
            'bagging_freq': hp.quniform('bagging_freq', 10, 500, 20),
            'learning_rate': hp.quniform('learning_rate', 0.01, 0.2, 0.01)}
    
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=5)
        
        print(best)
        
        best['boosting_type'] = 'gbdt'
        best['objective'] = 'quantile'
        best['alpha'] = 0.55 + 0.02 * need_for_holidays
        best['verbose'] = -1
        best['metrics'] = 'mae'
        for i in best:
            if type(best[i]) != str and type(best[i]) != set:
                if int(best[i]) == best[i]:
                    best[i] = int(best[i])
        
        threshold=round(X_train.shape[0] * 0.75)
        lgb_train = lgb.Dataset(X_train[:threshold], Y_train[:threshold])
        lgb_valid = lgb.Dataset(X_train[threshold:], Y_train[threshold:])
        
        del X_train, Y_train
        
        model = lgb.train(params=best, train_set=lgb_train, num_boost_round=500, valid_sets=lgb_valid, verbose_eval=False, early_stopping_rounds=4)
    
    
        model.save_model(model_path, num_iteration=model.best_iteration)
        
        cols = {}
        cols['place_id'] = df_func['place_id'].unique().tolist()
        cols['item_id'] = df_func['item_id'].unique().tolist()
        cols['train_cols'] = train_cols
        cols['all_cols'] = df_func.columns
        with open(columns_path, 'wb') as handle:
            pickle.dump(cols, handle)
        
    else:
    
        with open(columns_path, 'rb') as handle:
            all_dict = pickle.loads(handle.read())
        
        train_cols = all_dict['train_cols']
        all_cols = all_dict['all_cols']
        new_cols = list(set(all_cols)  - set(df_func.columns))
    
        for c in new_cols:
            df_func[c] = 0
            
        df_func = df_func[all_cols]
        model = lgb.Booster(model_file=model_path)
    return df_func, model, train_cols




# =============================================================================
# Делаем прогноз на следующие дни
# =============================================================================
def create_forecast(df_func, model, train_cols, predict_days, weeks_lag, days_lag, with_weather, last_date, abs_path):
   
    account = os.getenv('account')
    
    combined_path = os.path.join(abs_path, "data", "models", 'combined_params.txt')
    with open(combined_path, 'rb') as handle:
        combined_dict = pickle.loads(handle.read())
    if account in combined_dict:
        a, b = combined_dict[account]   
    else:
        a, b = 0, 0
    
    
    target='sales'
    pref = '_'
    # предсказанные данные
    future_df = pd.DataFrame()
    # реальные исторические данные
    lagged_df = df_func[(df_func['date'] >= last_date - datetime.timedelta(max(weeks_lag * 7, days_lag) + 2))&(df_func['date'] <= last_date)].copy()
    for d in range(predict_days):

        step_date = last_date + datetime.timedelta(d + 1)
        old_df = pd.concat([lagged_df, future_df], sort=False)
        new_df = df_func[df_func['date'] == (last_date + datetime.timedelta(days=1+d))].copy()
        new_df = pd.concat([old_df, new_df], sort=False)
        sleep(0.5)
        new_df = create_aggr.add_week_statistics(df_func = new_df, weeks_lag = weeks_lag)
        for new_lag in range(1, max(1, d+1)):
            date_i = 'date_' + str(new_lag) + 'ago'
            new_df[date_i] = new_df['date'].apply(lambda x: x - datetime.timedelta(days=new_lag))
            new_df.drop(target + pref + str(new_lag) + 'ago', axis=1, inplace=True)
            columns_right    = ['date','item_id', 'place_id', target]
            columns_left_on  = [date_i,'item_id', 'place_id']
            columns_right_on = ['date','item_id', 'place_id']
            new_df = new_df.merge(old_df[columns_right], left_on=columns_left_on, right_on=columns_right_on, suffixes=('', pref + str(new_lag) + 'ago'), how='left').drop_duplicates()
        # удаляем столбцы с лаговыми датами
        new_df.drop(columns = [i for i in new_df.columns if i.startswith('date') & i.endswith('ago')], axis=1, inplace=True)
        # отбираем столбцы с лаговыми значениями
        sales_columns = [i for i in new_df.columns if i.startswith(target + pref) & i.endswith('ago')]
        sales_week_columns  = [i for i in sales_columns if (int(re.findall(r'\d+', i)[0])<=7)]
        sales_week2_columns = [i for i in sales_columns if (int(re.findall(r'\d+', i)[0])<=14)]
        if days_lag > 21:
            new_df['w' + pref + 'avg_vtornik'] = new_df[[target + pref + '7ago', target + pref + '14ago', target + pref + '21ago']].mean(axis=1)
            new_df = create_aggr.add_statistics(new_df, sales_week_columns, 'w' + pref)
            sleep(0.5)
            new_df = create_aggr.add_statistics(new_df, sales_week2_columns, 'w2' + pref)
            new_df['w' + pref + 'ratio_1vtornik'] = new_df[target + pref + '1ago'] / new_df['w' + pref + 'avg_vtornik']
            new_df['w' + pref + 'ratio_2vtornik'] = new_df[target + pref + '2ago'] / new_df['w' + pref + 'avg_vtornik']
            new_df['w2' + pref + 'ratio_vtornik'] = new_df[target + pref + '7ago'] / new_df['w' + pref + 'avg_vtornik']
            new_df['w' + pref + 'ratio'] = new_df['w' + pref + 'sum'] / new_df['w2' + pref + 'avg']
            new_df['w' + pref + 'ratio_avg'] = new_df['w' + pref + 'avg'] / new_df['w2' + pref + 'avg']
        elif days_lag >= 14:
            new_df['w' + pref + 'avg_vtornik'] = new_df[[target + pref + '7ago', target + pref + '14ago']].mean(axis=1)
            new_df = create_aggr.add_statistics(new_df, sales_week_columns, 'w' + pref)
            new_df = create_aggr.add_statistics(new_df, sales_week2_columns, 'w2' + pref)
            new_df['w' + pref + 'ratio_1vtornik'] = new_df[target + pref + '1ago'] / new_df['w' + pref + 'avg_vtornik']
            new_df['w' + pref + 'ratio_2vtornik'] = new_df[target + pref + '2ago'] / new_df['w' + pref + 'avg_vtornik']
            new_df['w2' + pref + 'ratio_vtornik'] = new_df[target + pref + '7ago'] / new_df['w' + pref + 'avg_vtornik']
            new_df['w' + pref + 'ratio'] = new_df['w' + pref + 'sum'] / new_df['w2' + pref + 'avg']
            new_df['w' + pref + 'ratio_avg'] = new_df['w' + pref + 'avg'] / new_df['w2' + pref + 'avg']
        else:
            new_df['w' + pref + 'avg_vtornik'] = new_df[target + pref + '7ago']
        if with_weather:
            for i in range (1, 5):
                for var in ['temp', 'humidity', 'temp_min', 'temp_max', 'wind_speed', 'pressure']:
                    new_df[var + pref + str(i) +'ratio'] = new_df.apply(lambda x: x[target + pref + str(i) + 'ago'] if x[var + pref + str(i) + 'ago'] == 0 else x[var] / x[var + pref + str(i) + 'ago'] * x[target + pref + str(i) + 'ago'], axis=1)

        new_df.fillna(0, inplace=True)
        df_test  = new_df[new_df['date'] ==  step_date].copy()
        res = df_test[['date', 'place_id', 'item_id']]
        X_test = df_test[train_cols]
        Y = []
        if X_test.shape[0] > 0:
            Y = model.predict(X_test)
        res['sales_pred'] = Y
        new_df = new_df.merge(res, on=['date', 'place_id', 'item_id'], how='left')
        new_df.loc[:,target] = new_df.apply(lambda x: x['sales_pred'] if x[target]==0 else x[target], axis=1)
        new_df.drop(['sales_pred'], axis=1, inplace=True)
        future_df = pd.concat([future_df, new_df.loc[new_df['date'] == step_date]], sort=False)
        sleep(0.5)
    cols = ['sales_7ago', 'sales_14ago'] + ['sales_21ago'] * (weeks_lag > 2)
    # Turn lagged sales to absolute values    
    for col in cols:
        future_df[col + '_'] = 2 ** future_df[col] - 1        
    cols = [i + '_' for i in cols]
    #Calculate baseline and transform it into log
    mask = (future_df[cols].std(axis=1) / future_df[cols].mean(axis=1) < a/100) | (future_df[cols].mean(axis=1) < b) 
    baseline = future_df.loc[mask, cols].mean(axis=1)
    baseline = np.log2(baseline + 1)
    
    future_df.loc[mask, target] = baseline
    return future_df


#модель завышения
# =============================================================================
# Выгружаем дополнитльно завышенный прогноз для подсказки в зеленую подсветку
# =============================================================================
def create_help_highlight(df_func, with_expiry, df_cat_item_weekday):
    # корректирующее правило по списаниям
    if with_expiry:
        expiry_columns = [i for i in df_func.columns if i.startswith('expiry') & i.endswith('ago')]
        sales_columns = [i for i in df_func.columns if i.startswith('sales') & i.endswith('ago')]
        df_func = df_func[['date', 'place_id', 'item_id', 'sales'] + expiry_columns + sales_columns]
        exp_week_columns  = [i for i in expiry_columns if (int(re.findall(r'\d+', i)[0])%7 == 0)]
        sales_week_columns  = [i for i in sales_columns if (int(re.findall(r'\d+', i)[0])%7 == 0)]
        for col in ['sales'] + sales_columns + expiry_columns:
            df_func[col] = 2 ** df_func[col] - 1
        df_func['e_sum'] = df_func[expiry_columns].sum(axis=1)
        df_func['w_sum'] = df_func[sales_columns].sum(axis=1)
        df_func['e_avg'] = df_func[exp_week_columns].mean(axis=1)
        df_func['w_avg'] = df_func[sales_week_columns].mean(axis=1)
        df_func['delta'] = df_func.apply(lambda x: 1 if ((x['w_avg'] < 10) & (x['e_avg'] < 0.15 * x['w_avg']))
                                    else 3 if ((x['w_avg'] >= 10) & (x['e_sum']  <= 0.1 * x['w_avg']))
                                    else 2 if ((x['w_avg'] >= 10) & (x['e_avg']  <= 0.15 * x['w_avg']))
                                    else 1, axis=1)

        df_func[['delta']].groupby('delta').size().reset_index()
        df_func = df_func[df_func['date'] < (last_date + datetime.timedelta(days=7))]
        df_func['delta'] = df_func.apply(lambda x: x['delta'] if (x['e_sum'] > 0) else 1, axis=1)
        df_func['value'] = round(df_func['sales'] + df_func['delta'])
        df_func['dt'] = df_func['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df_func['prediction_type'] = 'order_high'
        df_func['model'] = model_name
        df_func['account_id'] = account
        df_func = df_func[['dt', 'value', 'prediction_type', 'model', 'item_id', 'place_id', 'account_id']]
        df_func.drop_duplicates(['place_id', 'dt', 'item_id'], inplace=True)
        data = df_func.to_dict('records')
        example_forecats = {'token': token, 'forecasts': data}
        r = requests.post(url = forecast_url, json = example_forecats)
        print(r.text)
        return df_func


def pred_step(df ,weather_url,with_expiry
            ,df_cat_item_weekday , temp_cat_df ,hum_cat_df , wind_speed_cat_df , price_cat_df , ura_day_cat_df
            ,model , model_columns):

    ##### делаем препроцессинг

    #add weather factors
    token = os.getenv('ML_API_KEY')

    df = gen_factors.add_calc_weather(df=df , weather_url=weather_url , token=token)
    ##### Логарифмируем и отбрасываем отрицательные продажи, если таковые имеются
    with_expiry = False
    df = check_info.values_transformation(df_func = df, with_expiry = with_expiry, columns = ['count'])
    #generate prev values
    prev_steps = 28
    prev_column = 'count'
    df , prev_columns = check_info.add_prev_vals(df,prev_steps  , prev_column)
    #generate steps statistics
    df , add_stats_columns = check_info.add_stats_info(pay_aggr_df = df , want_days_back = prev_steps , prev_columns = prev_columns)
    #add diff between periods
    df = check_info.add_diff_info(df = df , prev_columns = prev_columns)
    #add info about prev weeks
    df = check_info.add_prev_weeks_stats(df,prev_steps , prev_column)
    ###categorial features statistics
    df['weekday'] = pd.to_datetime(df['date']).dt.weekday
    #add aggr - weekdays stats
    #additional factor - prev_calc_1
    df = aggr_info.weekdays_stats_add(df , df_cat_item_weekday ,prev_column)
    #categoriser - temp info
    df = aggr_info.temp_categoriser(df)
    #add aggr - temp stats
    df = aggr_info.aggr_stats_add(df , aggr_info_df = temp_cat_df , aggr_column = 'temp_cat')
    #categoriser - hum info
    df = aggr_info.hum_categoriser(df)
    #add aggr - hum stats
    df = aggr_info.aggr_stats_add(df , aggr_info_df = hum_cat_df , aggr_column = 'hum_cat')
    #categoriser - wind_speed info
    df = aggr_info.wind_speed_categoriser(df)
    #add aggr - wind_speed stats
    df = aggr_info.aggr_stats_add(df , aggr_info_df = wind_speed_cat_df , aggr_column = 'wind_speed_cat')
    #categoriser - price info
    df = aggr_info.price_categoriser(df)
    #add aggr - price stats
    df = aggr_info.aggr_stats_add(df , aggr_info_df = price_cat_df , aggr_column = 'item_discount_cat')
    #categoriser - ura_day info
    df = aggr_info.uraday_categoriser(df)
    #add aggr - ura_day stats
    df = aggr_info.aggr_stats_add(df , aggr_info_df = ura_day_cat_df , aggr_column = 'ura_day')

    #делаем прогноз
    df['pred'] = model.predict(df[model_columns])

    return df


#v2 прогноз потребления
def main_v2(abs_path, df, with_expiry, predict_days):
    weather_url = '/api/ml/places/:place_id/weathers'
    count_column = 'sales'
    meta_columns = ['date','count','item_discount','category_id2','item_id2','place_id2']
    with_weather = True

    #load ids
    categories_df = pd.read_csv(abs_path + '/data_info/categories_df.csv' )
    items_df = pd.read_csv(abs_path + '/data_info/items_df.csv' )
    places_df = pd.read_csv(abs_path + '/data_info/places_df.csv' )

    #load stats
    df_cat_item_weekday = pd.read_csv(abs_path + '/aggr_info/df_cat_item_weekday.csv')
    temp_cat_df = pd.read_csv(abs_path + '/aggr_info/temp_cat_df.csv')
    hum_cat_df = pd.read_csv(abs_path + '/aggr_info/hum_cat_df.csv')
    wind_speed_cat_df = pd.read_csv(abs_path + '/aggr_info/wind_speed_cat_df.csv')
    price_cat_df = pd.read_csv(abs_path + '/aggr_info/item_discount_cat_df.csv')
    ura_day_cat_df = pd.read_csv(abs_path + '/aggr_info/ura_day_cat_df.csv')



    #load model columns
    with open(abs_path + '/models/model_columns.txt', "rb") as fp:
        model_columns = pickle.load(fp)

    #load lgbm regression model
    model = lgb.Booster(model_file=abs_path + '/models/lgbm_regression_model.txt')

    #load lgbm regression model
    model_high = lgb.Booster(model_file=abs_path + '/models/lgbm_regression_model_75perc.txt')


    ##### унифицируем для прогноза

    #change to normal ids
    df = pd.merge(df
                       ,categories_df
                       ,how = 'left'
                       ,left_on = 'category_id'
                       ,right_on = 'category_hash')


    df = pd.merge(df
                       ,items_df
                       ,how = 'left'
                       ,left_on = 'item_id'
                       ,right_on = 'item_hash')


    df = pd.merge(df
                       ,places_df
                       ,how = 'left'
                       ,left_on = 'place_id'
                       ,right_on = 'place_hash')

    #find missing items , places , categories
    df_missing = df[df.isnull().any(axis=1)][:]

    #rename count
    if count_column != 'count':
        df = df.rename(columns = {count_column:'count' })[:]

    #stay only meta columns
    df = df[meta_columns][:]

    #drop missings
    df.dropna(inplace = True)


    ###итеративно добавляем новые даты
    ###добавляем в датасет информацию о необходимых прогнозах
    last_date = df['date'].max()
    predict_dates = []

    for d in range(predict_days):
        new_date = last_date + timedelta(days=d+1)
        week_ago = new_date + timedelta(days=-7)

        df_week_ago = df[(df['date'] == week_ago)][:]
        df_week_ago['count'] = np.nan
        df_week_ago['date'] = new_date


        df = pd.concat([df,df_week_ago])

        predict_dates.append(new_date)

    df = df.reset_index(drop=True)[:]


    ###идем по дням и делаем прогнозы
    for d in predict_dates:

        df_pred = df[(df['date'] <= d)][:]

        df_pred = pred_step(df=df_pred ,weather_url=weather_url,with_expiry=with_expiry
            ,df_cat_item_weekday=df_cat_item_weekday , temp_cat_df=temp_cat_df ,hum_cat_df=hum_cat_df
            , wind_speed_cat_df=wind_speed_cat_df , price_cat_df=price_cat_df , ura_day_cat_df=ura_day_cat_df
            ,model=model,model_columns=model_columns)
        print(f'{d} pred done, shape {df_pred.shape[0]}')

        df = pd.merge(df
                ,df_pred[(df_pred['date'] == d)][['date','item_id2' , 'category_id2' , 'place_id2' , 'pred']]
                ,how = 'left'
                ,left_on = ['date','item_id2' , 'category_id2' , 'place_id2' ]
                ,right_on = ['date','item_id2' , 'category_id2' , 'place_id2' ]
                )[:]

        pred_index = df[~(df['pred'].isnull())].index

        df.loc[pred_index , 'count'] = np.round(df.loc[pred_index , 'pred'] , 3)

        df.drop(['pred'] , axis = 1 , inplace = True)




    ###оставляем прогнозы через 1 день для заливки
    df_pred = df[(df['date'] > last_date + timedelta(days=1))][:]


    df_pred = pd.merge(df_pred
             ,items_df
             ,how = 'left'
             ,left_on = ['item_id2']
             ,right_on = ['item_id2']
            )[:]

    df_pred = pd.merge(df_pred
             ,places_df
             ,how = 'left'
             ,left_on = ['place_id2']
             ,right_on = ['place_id2']
            )[:]


    df_pred.drop(['category_id2' , 'item_id2' , 'place_id2'] , axis = 1 , inplace = True)


    df_pred.rename(columns = {'item_hash' :'item_id' , 'place_hash':'place_id' , 'count':'sales'} , inplace = True)


    df_pred['Weekday'] = pd.to_datetime(df_pred['date']).dt.weekday


    return df_pred






