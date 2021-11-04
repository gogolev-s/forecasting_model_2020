import re
import datetime
import requests
import numpy as np
import pandas as pd
from time import sleep


def digits_round(df_func):
    settings = []
    for product in df_func.item_id.unique():
        df_temp = df_func[df_func['item_id']==product]
        digits = 0
        while (digits < 3) and (not
                                ((round(df_temp[['sales']], digits).equals(round(df_temp[['sales']], digits + 1))) &
                                (round(df_temp[['sales']], digits).equals(df_temp[['sales']])))
                                ):
            digits = digits + 1
        settings.append(pd.DataFrame([[product, digits]]))
    settings = pd.concat(settings, sort=False)
    settings.columns=['item_id', 'digits']
    settings['min'] = 0
    settings['plan'] = 0
    return settings


def sales_by_hours(df_func, rule_expiry):
    if rule_expiry:
        df_func = df_func[df_func['date']>df_func['date'].max() - datetime.timedelta(days=30)]
        by_hours = df_func.groupby(['hour'])['sales'].sum() / df_func.groupby(['hour'])['sales'].sum().sum()
        by_hours = by_hours.iloc[:-8:-1]
        by_hours = by_hours.cumsum().reset_index().rename(columns={'sales':'delta', 'hour':'hour_delta'})
    else:
        by_hours = pd.DataFrame()
    return by_hours


def transform_groups(df_func, base_url, items):
    groups_list = []
    for row, col in items.iterrows():
        children = pd.DataFrame(items.loc[row, 'children'])
        if children.shape[0] > 0:
            children = children[['ancestor_id', 'id', 'proportion', 'visible']]
            children['proportion'] = children['proportion'].astype('float32')
            children['name'] = items.loc[row, 'name']
            children['category'] = items.loc[row, 'category']
            groups_list.append(children)
    groups = pd.concat(groups_list)
    df_func = df_func.merge(groups, how='left', left_on = 'item_id', right_on='id')
    mask = ~ pd.isnull(df_func['id'])
    # Add rows with visible position
    new_rows = df_func[mask & (df_func['visible'])].copy()
    # Convert SKUs to groups according to info in "groups"
    if 'item_name' in df_func.columns:
        df_func.loc[mask, 'item_name'] = df_func.loc[mask, 'name']
    if 'item_category' in df_func.columns:
        df_func.loc[mask, 'item_category'] = df_func.loc[mask, 'category']
    df_func.loc[mask, 'item_id'] = df_func.loc[mask, 'ancestor_id']
    df_func.loc[mask, 'sales'] = df_func.loc[mask, 'proportion'] * df_func.loc[mask, 'sales']
    df_func = pd.concat([df_func, new_rows])
    df_func.drop(['ancestor_id', 'id', 'proportion', 'name', 'category', 'visible'], axis=1, inplace=True)
    return df_func


def create_df(df_func, rule_expiry, with_expiry=False, target_hour=0, expiry=None, last_trans_date=None, places=None):

    ##### If we decide to increase sales in days where there were no expiries
    if not rule_expiry:
        df_new = df_func.groupby(['date', 'item_id', 'place_id']).agg({'sales':'sum', 'item_discount':'median', 'item_name':'max'}).reset_index()
    else:
        df_new = df_func.groupby(['date', 'item_id', 'place_id']).agg({'sales':'sum', 'item_discount':'median', 'item_name':'max', 'hour':'max'}).reset_index()

    ##### Add some expiries as additional sales for prediction like food for staff
    if with_expiry:
        expiry = expiry[expiry['date'] <= last_trans_date]
        df_new = df_new.merge(expiry, how='left', on = ['date', 'item_id',  'place_id'])
        del expiry
        df_new['expiry'].fillna(0, inplace=True)
        df_new.loc[:,'sales'] = df_new['sales'] + df_new['expiry']
        expiry_sum = df_new.groupby(['item_id'])['expiry'].sum().reset_index().rename(columns={'expiry':'expiry_sum'})
        df_new.drop(['expiry', 'total'], axis=1, inplace=True)
        df_new = df_new.merge(expiry_sum, how='left', on = ['item_id'])

    df_new.drop_duplicates(['date','item_id', 'place_id'], inplace=True)
    return df_new


def create_df_hour(df_func, few_data=True):
    if few_data:
        df_new = df_func.groupby(['date', 'item_id', 'place_id', 'hour']).agg({'item_discount':'median', 'item_name':'max', 'sales':'sum'}).reset_index()
    else:
        df_new = df_func.groupby(['date', 'item_id', 'place_id']).agg({'item_discount':'median', 'item_name':'max'}).reset_index()
        min_hour = int(df_func['hour'].min())
        max_hour = int(df_func['hour'].max())
        hours = pd.DataFrame(list(range(min_hour, max_hour + 1))).rename(columns={0:'hour'})
        hours['key'] = 0
        df_new['key'] = 0
        df_new = df_new.merge(hours, on='key', how='left').drop('key', axis=1)
        df_temp = df_func.groupby(['date', 'item_id', 'place_id', 'hour']).agg({'sales':'sum'}).reset_index()
        df_new = df_new.merge(df_temp, on=['date', 'item_id', 'place_id', 'hour'], how='left')
        df_new.drop_duplicates(['date','item_id', 'place_id', 'hour'], inplace=True)
    return df_new

# =============================================================================
# Сохраняем данные по цене позиций и общему числу списаний за период
# =============================================================================
def create_nomenclature(df_func, with_expiry, last_trans_date, relevant_period = 30):
    products = df_func[df_func['date'] > last_trans_date -  datetime.timedelta(days=relevant_period)]
    group_vars = ['item_id', 'place_id'] +  ['sku'] * ('sku' in df_func.columns)
    if with_expiry:
        products = products.groupby(group_vars).agg({'item_discount':'median', 'item_name':'max', 'expiry_sum':'mean'}).reset_index()
    else:
        products = products.groupby(group_vars).agg({'item_discount':'median', 'item_name':'max'}).reset_index()

    return products



# =============================================================================
# Логарифмируем
# =============================================================================
def values_transformation(df_func, with_expiry, columns):
    for column in columns:
        df_func.loc[df_func[column] < 0, column] = 0
        df_func.loc[:, column] = round(np.log2(df_func[column] + 1), 2)
    return df_func


def rule_add_expiry(df_func, set_url, token_url, hours_dict, expiry_rule, add_remainders):
    #    df_func=df
    remainders = pd.DataFrame()
    if add_remainders:
        for place in df_func.place_id.unique():
            rem_url = set_url + place + '/items_balances?' + token_url
            rem = pd.DataFrame.from_dict(requests.get(rem_url).json()['items_balances'])
            rem['place_id'] = place

            if rem.shape[0] > 0:
                remainders = pd.concat([remainders, rem], sort=True)
        remainders.rename(columns={'amount':'remainders'}, inplace=True)
        remainders['date'] = remainders['dt'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
        remainders = remainders[['place_id', 'item_id', 'date', 'remainders']]
        expiry_rule = expiry_rule.merge(remainders, on=['place_id', 'item_id', 'date'], how='outer').fillna(0)
        expiry_rule['expiry'] = expiry_rule['expiry'] + expiry_rule['remainders']
    df_func = df_func.merge(expiry_rule[['date', 'place_id', 'item_id', 'expiry']], on=['date', 'place_id', 'item_id'], how='left')
    df_func = df_func.merge(hours_dict, left_on=['hour'], right_on=['hour_delta'], how='left')
    df_func.delta.fillna(hours_dict.delta.max(), inplace=True)
    #To avoid nans exclusion
    mask = ~(df_func['expiry'] > 0) & (df_func['item_id'].isin(expiry_rule.item_id.unique()))
    df_func.loc[mask, 'sales'] = df_func.loc[mask, 'sales'] * (1 + 0.65 * df_func.loc[mask, 'delta'])
    df_func.drop(['hour', 'hour_delta', 'delta', 'expiry'] , axis=1, inplace=True)
    return df_func, expiry_rule

# =============================================================================
# Добавляем статистики в тот же день/час ровно неделю назад
# =============================================================================
def add_week_statistics(df_func, weeks_lag, additional_key=[], features=['sales']):
    weeks_lag = max(weeks_lag, 2)
    if  len(additional_key) > 0:
        pref = '_h_'
    else:
        pref = '_'
    initial_columns = [i for i in df_func.columns if i.startswith('sales' + pref) & i.endswith('ago')]
    if len(initial_columns) == 0:
        replace = False
    else:
        temp_df = df_func[['date', 'place_id', 'item_id'] + initial_columns + additional_key].copy()
        df_func.drop(initial_columns, inplace=True, axis=1)
        replace = True

    # Получаем данные по продажам за предыдущие 7, 14, ..., 7*lag дней
    for i in range(7, (weeks_lag + 1) * 7, 7):
        date_i = 'date_' + str(i) + 'ago'
        df_func[date_i] = df_func['date'].apply(lambda x: x + datetime.timedelta(days=i))
        columns_right    = [date_i,'item_id', 'place_id'] + features + additional_key
        columns_left_on  = ['date','item_id', 'place_id'] + additional_key
        columns_right_on = [date_i,'item_id', 'place_id'] + additional_key
        df_func = df_func.merge(df_func[columns_right], left_on=columns_left_on, right_on=columns_right_on, suffixes=('', pref + str(i) + 'ago'), how='left').drop(date_i + pref + str(i) + 'ago', axis=1).drop_duplicates()
        sleep(0.5)

    df_func.fillna(0, inplace=True)

    # удаляем столбцы с лаговыми датами
    df_func.drop(columns = [i for i in df_func.columns if i.startswith('date') & i.endswith('ago')], axis=1, inplace=True)
    for var in features:
        week_stat_columns = [i for i in df_func.columns if i.startswith(var + pref) & i.endswith('ago')]
        df_func['weeks_' + var + pref + 'ago_mean'] = df_func[week_stat_columns].mean(axis=1)
        df_func['weeks_' + var + pref + 'ago_min'] = df_func[week_stat_columns].min(axis=1)
        df_func['week_' + var + pref + 'ago_max'] = df_func[week_stat_columns].max(axis=1)
        df_func['week_' + var + pref + 'ago_std'] = df_func[week_stat_columns].std(axis=1)
        df_func['week_' + var + pref + 'ago_trend'] = df_func.apply(lambda x: np.polyfit(np.arange(weeks_lag), list(x[week_stat_columns]), 1)[0], axis=1)
        df_func['weeks_' + var + pref + 'ago_dev'] =  df_func.apply(lambda x: x[var + pref + '7ago'] / x['weeks_' + var + pref + 'ago_mean'] if x['weeks_' + var + pref + 'ago_mean'] > 0 else 1, axis=1)
        df_func['weeks_' + var + pref + 'ago_miss'] = (df_func[week_stat_columns] == 0).sum(axis=1)
        #new_columns = ['weeks_' + var + 'ago_mean', 'weeks_' + var + pref + 'ago_min', 'weeks_' + var + pref + 'ago_max', 'weeks_' + var + pref + 'ago_std', 'weeks_' + var + pref + 'ago_trend', 'weeks_' + var + pref + 'ago_dev', 'weeks_' + var + pref + 'ago_miss']
        if var =='sales_before':
             df_func['weeks_before_ratio_day'] = df_func['weeks_' + var + pref + 'ago_mean'] / df_func['day_sales']
             #df_func['weeks_before_ratio_mor'] = df_func['weeks_' + var + pref + 'ago_mean'] / df_func['morning_sales']

        df_func.drop(week_stat_columns, axis=1, inplace=True)
    if replace:
        df_func = df_func.merge(temp_df, on=['date', 'place_id', 'item_id'] + additional_key, how='left')

    return df_func

# =============================================================================
# Статистики для продаж, списаний и погоды
# =============================================================================
def add_statistics(df0, cols, col_name):
    df0[col_name + 'sum'] = df0[cols].sum(axis=1)
    df0[col_name + 'avg'] = df0[cols].mean(axis=1)
    df0[col_name + 'std'] = df0[cols].std(axis=1)
    df0[col_name + 'q90'] = df0[cols].quantile(q=0.90,interpolation = 'lower', axis=1)
    df0[col_name + 'q50'] = df0[cols].quantile(q=0.50,interpolation = 'lower', axis=1)
    df0[col_name + 'q75'] = df0[cols].quantile(q=0.75,interpolation = 'lower', axis=1)
    return df0


# =============================================================================
# Добавляем лаговые значения переменных с макс лагом  days_lag
# =============================================================================
def add_lagged_sales(df_func, days_lag, with_weather=True, additional_key=[]):
    features = ['sales']
    if len(additional_key)>0:
        pref = '_h_'
    else:
         pref = '_'


    for i in range(1, days_lag + 1):
        date_i = 'date_' + str(i) + 'ago'
        df_func[date_i] = df_func['date'].apply(lambda x: x + datetime.timedelta(days=i))
        # with_expiry и with_weather - бинарные переменные
        columns_right    = [date_i,'item_id', 'place_id'] + features + ['temp', 'temp_min', 'temp_max', 'humidity', 'wind_speed', 'pressure'] * with_weather + additional_key
        columns_left_on  = ['date','item_id', 'place_id'] + additional_key
        columns_right_on = [date_i,'item_id', 'place_id'] + additional_key
        sleep(0.2)
        df_func = df_func.merge(df_func[columns_right], left_on=columns_left_on, right_on=columns_right_on, suffixes=('', pref + str(i) + 'ago'), how='left').drop(date_i + pref + str(i) + 'ago', axis=1).drop_duplicates()

    df_func.fillna(0 , inplace = True)
    df_func.replace({float('NaN'): 0}, inplace=True)
    # удаляем столбцы с лаговыми датами
    df_func.drop(columns = [i for i in df_func.columns if i.startswith('date') & i.endswith('ago')], axis=1, inplace=True)
    # отбираем столбцы с лаговыми значениями
    for var in features:
        sales_columns = [i for i in df_func.columns if i.startswith(var + pref) & i.endswith('ago')]
        # лаговые продажи
        sales_week_columns  = [i for i in sales_columns if (int(re.findall(r'\d+', i)[0])<=7)]
        sales_week2_columns = [i for i in sales_columns if (int(re.findall(r'\d+', i)[0])<=14)]
        # считаем статистики и вручную созданные отношения к тому же дню прошлой неделе
        if days_lag > 21:
            df_func['w' + pref + 'avg_vtornik'] = df_func[[var + pref + '7ago', var + pref + '14ago', var + pref + '21ago']].mean(axis=1)
            df_func = add_statistics(df_func, sales_week_columns, 'w' + pref)
            df_func = add_statistics(df_func, sales_week2_columns, 'w2' + pref)
            df_func['w' + pref + 'ratio_1vtornik'] = df_func[var + pref + '1ago'] / df_func['w' + pref + 'avg_vtornik']
            df_func['w' + pref + 'ratio_2vtornik'] = df_func[var + pref + '2ago'] / df_func['w' + pref + 'avg_vtornik']
            df_func['w2' + pref + 'ratio_vtornik'] = df_func[var + pref + '7ago'] / df_func['w' + pref + 'avg_vtornik']
            df_func['w' + pref + 'ratio'] = df_func['w' + pref + 'sum'] / df_func['w2' + pref + 'avg']
            df_func['w' + pref + 'ratio_avg'] = df_func['w' + pref + 'avg'] / df_func['w2' + pref + 'avg']
        elif days_lag >= 14:
            df_func['w' + pref + 'avg_vtornik'] = df_func[[var + pref + '7ago', var + pref + '14ago']].mean(axis=1)
            df_func = add_statistics(df_func, sales_week_columns, 'w' + pref)
            df_func = add_statistics(df_func, sales_week2_columns, 'w2' + pref)
            df_func['w' + pref + 'ratio_1vtornik'] = df_func[var + pref + '1ago'] / df_func['w' + pref + 'avg_vtornik']
            df_func['w' + pref + 'ratio_2vtornik'] = df_func[var + pref + '2ago'] / df_func['w' + pref + 'avg_vtornik']
            df_func['w2' + pref + 'ratio_vtornik'] = df_func[var + pref + '7ago'] / df_func['w' + pref + 'avg_vtornik']
            df_func['w' + pref + 'ratio'] = df_func['w' + pref + 'sum'] / df_func['w2' + pref + 'avg']
            df_func['w' + pref + 'ratio_avg'] = df_func['w' + pref + 'avg'] / df_func['w2' + pref + 'avg']
        else:
            df_func['w' + pref + 'avg_vtornik'] = df_func[var + pref + '7ago']


    # статистики для погоды
    if with_weather:
        add_lagged_features = ['temp', 'temp_min', 'temp_max', 'humidity', 'wind_speed', 'pressure']
    else: add_lagged_features = []
    new_columns = []
    for var in add_lagged_features:
        temp_columns = [i for i in df_func.columns if i.startswith(var + pref) & i.endswith('ago')]
        temp_week_columns  = [i for i in temp_columns if (int(re.findall(r'\d+', i)[0])<=7)]
        temp_week2_columns = [i for i in temp_columns if (int(re.findall(r'\d+', i)[0])<=14)]
        df_func = add_statistics(df_func, temp_week_columns, var + pref)
        df_func = add_statistics(df_func, temp_week2_columns, var + '2' + pref)
        if days_lag >= 14: df_func[var + pref + 'ratio'] = df_func[var + pref + 'sum'] / df_func[var + '2' + pref + 'sum']
        new_columns.extend(temp_week2_columns)
    # вручную созданные переменные изменения погоды (s0 * t0/t1)
    if with_weather:
        for var in features:
            for i in range (1, 4):
                for add_var in add_lagged_features:
                    df_func[add_var + pref + str(i) +'ratio'] = df_func.apply(lambda x: x[var + pref + str(i) + 'ago'] if x[add_var + pref + str(i) + 'ago'] == 0 else x[add_var] / x[add_var + pref + str(i) + 'ago'] * x[var + pref + str(i) + 'ago'], axis=1)

    df_func['day'] = df_func['date'].apply(lambda x: x.day)
    df_func['is_zp_date'] = df_func['day'].apply(lambda x: 1 if x in [8, 9, 22, 23, 24] else 0)


    df_func.loc[df_func['sales'] < 0, 'sales'] = 0

    return df_func.drop_duplicates()


# =============================================================================
# Добавляем дамми переменные
# =============================================================================
def add_dummies(df_func, x_dummies):
    # Создаём дамми-переменные
    df_func = pd.merge(df_func, pd.get_dummies(df_func[x_dummies], columns=x_dummies), how='left', on=df_func.index, right_index=False).drop('key_0', axis=1)
    return df_func.drop_duplicates()


#df_func = future_df
def bushe_strange_rule(df_func, expiry_rule):
    df_func = df_func[['date', 'sku', 'sales', 'item_id', 'place_id']]
    expiry_rule.drop_duplicates(['item_id', 'place_id', 'date'], inplace=True)
    expiry_rule = expiry_rule[expiry_rule['date'] > df_func['date'].min() - datetime.timedelta(days=14)]
    expiry_date = min(df_func.date.min() - datetime.timedelta(days=1), expiry_rule.date.max())

    df_temp = df_func
    exp_temp = expiry_rule[['item_id', 'place_id']].drop_duplicates()
    exp_temp['value'] = 1
    df_temp = df_temp.merge(exp_temp, on=['item_id', 'place_id'], how='left')
    df_func = df_temp[~(df_temp['value'] > 0)].drop('value', axis=1)
    df_temp = df_temp[df_temp['value'] > 0].drop('value', axis=1)

    expiry_rule['date_2'] =  expiry_rule['date'] + datetime.timedelta(days=2)
    expiry_rule['date_3'] =  expiry_rule['date'] + datetime.timedelta(days=3)
    expiry_rule['date_7'] =  expiry_rule['date'] + datetime.timedelta(days=7)


    df_temp = df_temp.merge(expiry_rule[['item_id', 'place_id', 'date_2', 'expiry']].rename(columns={'expiry':'expiry_2'}), left_on=['date', 'item_id', 'place_id'], right_on=['date_2', 'item_id', 'place_id'], how='left')
    df_temp = df_temp.merge(expiry_rule[['item_id', 'place_id', 'date_3', 'expiry']].rename(columns={'expiry':'expiry_3'}), left_on=['date', 'item_id', 'place_id'], right_on=['date_3', 'item_id', 'place_id'], how='left')
    df_temp = df_temp.merge(expiry_rule[['item_id', 'place_id', 'date_7', 'expiry']].rename(columns={'expiry':'expiry_7'}), left_on=['date', 'item_id', 'place_id'], right_on=['date_7', 'item_id', 'place_id'], how='left')

    df_temp.loc[df_temp['date'] > expiry_date + datetime.timedelta(days=1+2), 'expiry_2'] = 1
    df_temp.loc[df_temp['date'] > expiry_date + datetime.timedelta(days=1+3), 'expiry_3'] = 1
    df_temp.loc[df_temp['date'] > expiry_date + datetime.timedelta(days=1+7), 'expiry_7'] = 1

    df_temp['delta'] = 4 - (df_temp[['expiry_2', 'expiry_3']] > 0).sum(axis=1) - 2 * (df_temp['expiry_7'] > 0)
    df_temp['diff'] = 0
    mask = df_temp['sales'] <= 5
    df_temp.loc[mask, 'diff'] = (df_temp[mask]['delta'] >= 2) * 1
    mask = (df_temp['sales'] > 5) & (df_temp['sales'] < 50)
    df_temp.loc[mask, 'diff'] = df_temp[mask].apply(lambda x: x['sales'] * (0.075 * x['delta']), axis=1)
    mask = df_temp['sales'] >= 50
    df_temp.loc[mask, 'diff'] = df_temp[mask].apply(lambda x: x['sales'] * (0.05 * x['delta']), axis=1)
    df_temp['sales'] = df_temp['sales'] + df_temp['diff']

    df_temp = df_temp[['date', 'sku', 'sales', 'item_id', 'place_id']]
    df_func = pd.concat([df_func, df_temp], sort=True)

    return df_func
