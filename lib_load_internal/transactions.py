import os
import pandas as pd
import datetime
from pytz import timezone
import requests
from time import sleep


# =============================================================================
# Загрузка транзакций через аккаунт
# =============================================================================
def download_transactions(trn_url, abs_path, first_date, last_trans_date, places, items, period_load=2):
    start_time = datetime.datetime.today()
    first_date_load = first_date
    transactions = pd.DataFrame()
    transactions_list = []
    file_path = os.path.join(abs_path, "data", "transactions", os.getenv('account') + '.csv')
    if os.path.exists(file_path):

        trans_temp = pd.read_csv(file_path, sep=',', decimal='.', parse_dates=[0], infer_datetime_format=True, low_memory=False)
        trans_temp.loc[:,'date'] = trans_temp['date'].apply(lambda x: x.date())
        trans_temp = trans_temp[['date', 'category_id', 'item_id', 'place_id', 'sales', 'hour', 'external_id', 'amount', 'item_discount']]
        #if saved data is enough BUT reload once a month
        random_int = ord(os.getenv('account')[0])
        #RELOAD ALL TRANSACTIONS DATA
        if (first_date >= trans_temp['date'].min()) & (((random_int + 5) % 13) != (last_trans_date.day % 13) | (os.name == 'nt')):
            #rewrite the last day in csv for sure
            first_date_load = trans_temp['date'].max()
            trans_temp = trans_temp[trans_temp['date'] < first_date_load]
            transactions_list = [trans_temp]
        elif first_date < trans_temp['date'].min():
            first_date_load = first_date
            trans_temp = trans_temp[trans_temp['date'] < first_date_load]
            transactions_list = [trans_temp]
        del trans_temp

    # Тащим все транзакции из аккаунта периодами по 2 дня
    period = max(0, (last_trans_date - first_date_load).days)
    for delta in range((period + 1)//period_load + 1):
        from_date = first_date_load + datetime.timedelta(days=delta*period_load)
        to_date   = min(first_date_load + datetime.timedelta(days=delta*period_load + period_load), last_trans_date +  datetime.timedelta(days=1))
        transactions_url = trn_url + '&from=' + str(from_date) + '&to=' + str(to_date)
        temp_request = requests.get(transactions_url)
        temp = temp_request.json()['transactions']
        temp = pd.DataFrame.from_dict(temp)
        sleep(1)
        if temp.shape[0] > 0:
            # Заменяем пропуски пустыми значениями
            temp = temp.merge(places[['place_id', 'tz']], how='inner', on=['place_id'])
            temp['category_id'].replace({None:''}, inplace=True)
            temp['date'] = temp[['dt', 'tz']].apply(lambda x: datetime.datetime.strptime(x['dt'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone('UTC')).astimezone(timezone(x['tz'])).date(), axis=1)
            temp['hour'] = temp[['dt', 'tz']].apply(lambda x: datetime.datetime.strptime(x['dt'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone('UTC')).astimezone(timezone(x['tz'])).hour, axis=1)
            # Формируем продажи  и цену за единицу
            temp['sales'] = temp['amount']
            temp = temp[['date', 'category_id', 'item_id', 'place_id', 'sales', 'hour', 'external_id', 'amount', 'item_discount']]
            transactions_list.append(temp)

    transactions = pd.concat(transactions_list, ignore_index=True, sort=False)
    #transactions.drop_duplicates(inplace=True)
    transactions.to_csv(file_path, sep=',', decimal='.', index=False)

    print('Transactions load: ', (datetime.datetime.today() - start_time).seconds//60, ' minutes')
    transactions = transactions[(transactions['date'] <= last_trans_date)&(transactions['date'] >= first_date)]

    transactions = transactions.merge(items[['id', 'name', 'category']], left_on='item_id', right_on='id', how='left')
    transactions = transactions.rename(columns={'name':'item_name', 'category':'item_category'})
    transactions.drop('id', axis=1, inplace=True)

    for col in ['amount', 'sales', 'hour', 'item_discount']:
        transactions[col].fillna(0, inplace=True)
        transactions[col] = transactions[col].astype(float)

    transactions[['item_name','item_category']].fillna('', inplace=True)
    transactions['item_name'] = transactions['item_name'].astype(str)
    transactions['item_category'] = transactions['item_category'].astype(str)

    transactions.dropna(subset=['place_id', 'item_id'], inplace=True)
    transactions.fillna('', inplace=True)

    return transactions



# =============================================================================
# Загрузка списаний через аккаунт
# =============================================================================
def download_writeoffs(expiry_reason, wrt_url, first_date, last_trans_date, places, rule_expiry, period_load=10):
    start_time = datetime.datetime.today()
    period = (last_trans_date - first_date).days
    expiry_rule_reason = [] + rule_expiry * ['expiry']
    expiry_rule = pd.DataFrame()
    expiry_add_to_df = pd.DataFrame()
    for delta in range((period + 1)//period_load + 1):
        from_date = first_date + datetime.timedelta(days=delta*period_load)
        to_date   = min(first_date + datetime.timedelta(days=delta*period_load + period_load), last_trans_date +  datetime.timedelta(days=1))
        transactions_url = wrt_url + '&from=' + str(from_date) + '&to=' + str(to_date)
        temp_request = requests.get(transactions_url)
        temp = temp_request.json()['write_offs']
        temp = pd.DataFrame.from_dict(temp)
        sleep(0.1)
        if temp.shape[0] > 0:
            temp = temp.merge(places[['place_id', 'tz']], how='inner', on=['place_id'])
            temp['date'] = temp[['dt', 'tz']].apply(lambda x: datetime.datetime.strptime(x['dt'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone('UTC')).astimezone(timezone(x['tz'])).date(), axis=1)
            temp['expiry'] = temp['amount']
            if len(expiry_reason) > 0:
                temp_exp = temp[temp['kind'].isin(expiry_reason)]
                temp_exp = temp_exp.groupby(['date',  'item_id', 'place_id']).agg({'expiry':'sum', 'total':'sum', 'cost_price':'median'}).reset_index()
                expiry_add_to_df = pd.concat([expiry_add_to_df, temp_exp], ignore_index=True, sort=False)
            if len(expiry_rule_reason) > 0:
                temp_rule = temp[temp['kind'].isin(expiry_rule_reason)]
                temp_rule = temp_rule.groupby(['date',  'item_id', 'place_id']).agg({'expiry':'sum'}).reset_index()
                expiry_rule = pd.concat([expiry_rule, temp_rule], ignore_index=True, sort=False)

    print('Exriry load: ', (datetime.datetime.today() - start_time).seconds//60, ' minutes')

    return expiry_add_to_df, expiry_rule