# -*- coding: utf-8 -*-
import os
import sys
import datetime
import pickle
import pandas as pd
import lightgbm as lgb
from dotenv import load_dotenv

if os.name == 'nt':
    abs_path = 'E:/Git/forecast_meta_models_test/src'
    os.chdir(abs_path)
else:
    abs_path = os.path.dirname(os.path.abspath(__file__))

load_dotenv(dotenv_path=os.path.join(abs_path, "data", 'info.env'))

##### Set variables
base_url = os.getenv('ML_API_BASE_URL')
token = os.getenv('ML_API_KEY')
##### API token
token_url = 'token=' + token

pd.options.mode.chained_assignment = None

from lib_load_internal import places as load_places
from lib_load_internal import items as load_items
from lib_load_internal import transactions as load_transactions
#from lib_load_internal import weather as load_weather
from lib_preprocess import create_kpi_analytics
from lib_preprocess import create_aggr, add_holidays
from lib_forecast import kpi_forecast, kissgbm as kgbm
from lib_save import save_results



##### Load info about all clients
with open(os.path.join(abs_path, "clients_info.txt"), 'rb') as handle:
    client_params = pickle.loads(handle.read())

client = sys.argv[1]

#client = 'Zelenograd' #### For testing
client_pars = client_params[client]

expiry_reason = []
include_today = False
day_after_today = 1

account = client_pars['account']
dummies = client_pars['dummies']
training_weekday = client_pars['training_weekday']
first_transaction = client_pars['first_transaction']
rule_expiry, add_remainders, smart_rounding  = False, False, False

if 'expiry_reason' in client_pars: expiry_reason = client_pars['expiry_reason']
if 'day_after_today' in client_pars: day_after_today = client_pars['day_after_today']
if 'rule_expiry' in client_pars: rule_expiry = client_pars['rule_expiry']
if 'add_remainders' in client_pars: add_remainders = client_pars['add_remainders']
if 'smart_rounding' in client_pars: smart_rounding = client_pars['smart_rounding']

min_retail_amount = 0

##### urls setup
places_url = base_url + 'accounts/' + account + '/places?' + token_url
set_url = base_url + 'places/'
forecast_url = base_url + 'forecasts/items'
forecast_plan_url = base_url + 'forecasts/points'
plans_url = base_url + 'accounts/' + account + '/plans?' + token_url
trn_url = base_url + 'accounts/' + account + '/transactions?' + token_url
wrt_url = base_url + 'accounts/' + account + '/write_offs?' + token_url
mlt_url = base_url + 'accounts/' + account + '/items_points?'  + token_url
plans_url = base_url + 'accounts/' + account + '/plans?' + token_url
items_url = base_url + 'accounts/' + account + '/items?' + token_url

model_name = 'kgbm_1'
with_expiry = len(expiry_reason) > 0
with_weather = False
target_hour = 0

os.environ["account"] = account
os.environ["model_name"] = model_name

##### parameters on dates to load
valid_weeks = 1 # model uses last week of data to validate model selection
event_weeks = 18 # model uses maximum 18 weeks of events to create models
feats_weeks = 5 # each event uses another 5 weeks for feature engineering -- note, this does not impact f.e. automatically as of now, need to add that functionality later

predict_days = 16 # how many days ahead to predict, t+1, t+2, etc..

##### The last date with transactions, + determine fcast_date
last_trans_date = datetime.date.today() - datetime.timedelta(days=1 - include_today)
fcast_date = last_trans_date + datetime.timedelta(days=1)

##### If there is a holiday we load more data
holidays_dict, need_for_holidays, train_model_today = add_holidays.holidays(last_trans_date, training_weekday, abs_path)
if need_for_holidays:
    event_weeks = 40
period = 7 * valid_weeks + 7 * event_weeks + 7 * feats_weeks
##### Threshold date for data loading
first_trans_date = last_trans_date - datetime.timedelta(days=period)
##### Date of first ransaction by the client in order to avoid empty requests
first_trans_date = max(first_trans_date, first_transaction)

#### TODO =====================================================================
'''
Add logic to constraint event_weks if first_trans_date < first_transaction.
We need to have at least 3 event_weeks with the full 5 feats_weeks present, otherwise FAIL, shouldn't model!
This means we need a min of 7 (val) + 21 (events) + 35 (features) = 63 days of transactions.
In the future we can also constraint a little bit feats_weeks to minumum = 4, so minimum data for any client is just 2 months.
'''
#### TODO =====================================================================

# =============================================================================
# load transactions
# =============================================================================
places = load_places.load(places_url)
items, items_group = load_items.load(items_url)

transactions = load_transactions.download_transactions(trn_url, abs_path, first_trans_date, last_trans_date, places, items)
print('===> transactions load')

##### Keep only active places
places = places[places['active']]
transactions = transactions[transactions['place_id'].isin(places['place_id'].unique())]

last_trans_date = min(last_trans_date, transactions.date.max())
fcast_date = last_trans_date + datetime.timedelta(days=1)
##### Send KPI once a month
if last_trans_date.day <= 2:
    bill_count, avg_bill = create_kpi_analytics.create_analytics_dfs(df_func=transactions)
    x = kpi_forecast.predict_analytics(forecast_plan_url, plans_url, bill_count, avg_bill, last_trans_date)

##### Keep only retail sales
if min_retail_amount > 0: transactions = transactions[transactions['sales'] <= min_retail_amount]

##### Calculate actual price = item_discount
transactions.loc[:, 'item_discount'] = transactions['item_discount'] / (transactions['amount'])
transactions.drop(['external_id', 'amount'], axis=1, inplace=True)

if smart_rounding: settings = create_aggr.digits_round(transactions)

##### Transformation some groups if they exist
if items_group.shape[0] > 0:
    transactions = create_aggr.transform_groups(transactions, base_url, items_group)

##### Calculate how much is sold at each hour to recaluate sales in days without expiries
hours_dict = create_aggr.sales_by_hours(transactions, rule_expiry)

##### Transform transactions to daily df
expiry = None
if with_expiry or rule_expiry:
    expiry, expiry_rule = load_transactions.download_writeoffs(expiry_reason, wrt_url, first_trans_date, last_trans_date, places, rule_expiry)
df = create_aggr.create_df(transactions, rule_expiry, with_expiry, target_hour, expiry, last_trans_date, places)

del transactions

if rule_expiry:
    df, expiry_rule = create_aggr.rule_add_expiry(df, set_url, token_url, hours_dict, expiry_rule, add_remainders)

df = df[df['sales'] > 0] # remove no sales data, if ever happens


df = add_holidays.add_holidays_factors(df, holidays_dict)
##### New weather loading

#cities = {}
#weather = pd.DataFrame()
#for index, row in places.iterrows():
#      cities[row['city_id']] = (row['tz'], row['city'])
#for city_id in cities:
#    weather_temp = load_weather.load_weather(city_id, cities[city_id][0], abs_path, base_url, token_url, last_trans_date)
#    weather_temp['city'] = cities[city_id][1]

##### Some clients have places in different cities
#weather = pd.concat([weather, weather_temp], sort=False)
#if weather.shape[0] > 0:
#    df = df.merge(places[['place_id', 'city']], on='place_id', how='left')
#    df = df.merge(weather, on=['date', 'city'], how='left')
#    df.drop('city', axis=1, inplace=True)

'''
temp_avg - average temperature during the day
temp_min - mininmun temperature during the day (equals to average in forecasting after 5 days)
temp_max - maximum temperature during the day (equals to average in forecasting after 5 days)
pressure - average pressure during the day
humidity - average humidity during the day
wind_speed - average wind speed during the day
precipitation - from 0 to 24, it shows in what number of hours precipitations were. 24 for absolutely rainy day, for example.
'''

#### Now we do not use weather so remove it


# =============================================================================
# Ivan's old code
# =============================================================================

df['sku'] = df['place_id'] + df['item_id'] # sku unique id feature
print('===> df daily create')

##### Keep table with cost and prices and combination of sku and item/place id in order to get it back in the end
products = create_aggr.create_nomenclature(df, with_expiry, last_trans_date)

df.drop(['item_id', 'place_id', 'item_name', 'item_discount'] + with_expiry * ['expiry_sum', 'cost_price'], axis=1, inplace=True)
#df = df[['date', 'sku', 'sales']]
# =============================================================================
# kissgbm create best model
# =============================================================================

# save original df for later use when forecasting and keep only used features with proper name
df.drop_duplicates(['date', 'sku'], inplace=True)
df_ori = df.copy()
df = df.rename(columns={'sales':'y'})
#df = df[['date', 'sales', 'sku']].rename(columns={'sales':'y'})
#df = df.rename(columns={'sales':'y'})


# create features kgbm way
fe_cutoff_date = fcast_date - datetime.timedelta(days=7 * valid_weeks + 7 * event_weeks) # set the first date of data to be used after f.e. is done
df = kgbm.create_features(df, fe_cutoff_date) # create features with cutoff date selection

# set validation dates
valid_last = fcast_date - datetime.timedelta(days=1)
valid_first = fcast_date - datetime.timedelta(days=7 * valid_weeks)

# run booster grid
params_list = kgbm.params_list() # parameters to evaluate
valid_grid = kgbm.valid_grid(df, params_list, valid_last, valid_first)

# select best booster found according to validation grid
best_valid = valid_grid.sort_values('swaa_va', ascending=False).iloc[0]
best_booster = lgb.Booster(model_str=best_valid['booster_string'], silent=True)

# =============================================================================
# kissgbm recursive forecasting, steps while iterating predict_days:
    # determine current step_fcast_date as fcast_date + step
    # use df_ori to grab skus from past weekday (delta 7 days) and build fc_df with current forecast date
    # get all those skus past data and do f.e. keeping only the current forecast date after
    # do prediction and append results to df_ori, which will append current forecast
    # repeat with step + 1, now we have last forecast in df_ori to be recursive
# =============================================================================

for step in range(predict_days):
    step_fcast_date = fcast_date + datetime.timedelta(days=step)

    # grab skus that were present the past weekday and append past data for them only
    fc_df = df_ori[df_ori['date'] == step_fcast_date - datetime.timedelta(days=7)][['sku']].copy()
    fc_df['date'] = step_fcast_date # create forecast date column
    sku_list = fc_df['sku'].unique() # get list of skus to forecast
    fc_df = fc_df.append(df_ori[df_ori['sku'].isin(sku_list)], sort='True')

    # f.e.
    fc_df = fc_df[['date', 'sales', 'sku']].rename(columns={'sales':'y'})
    if fc_df.shape[0] == 0:
        continue
    fc_df = kgbm.create_features(fc_df, step_fcast_date).drop(columns='y') # now this cutoff will always yield only 1 date, the one we are forecasting
    fc_df = add_holidays.add_holidays_factors(fc_df, holidays_dict)
    #fc_df = fc_df.merge(products[['place_id', 'sku']], on='sku', how='left')
    #fc_df = fc_df.merge(places[['place_id', 'city']], on='place_id', how='left')
    #fc_df = fc_df.merge(weather, on=['date', 'city'], how='left')
    #fc_df.drop(['place_id', 'city'], axis=1, inplace=True)

    # predict and round
    fc_df['y'] = best_booster.predict(fc_df[best_booster.feature_name()])

    # save forecasts to original for next iteration
    df_ori = df_ori.append(fc_df[['date', 'sku', 'y']].rename(columns={'y':'sales'}), sort='True')


# select forecasts
future_df = df_ori[df_ori['date'] >= fcast_date][['date', 'sku', 'sales']].copy()

# merge product data, this now only gets item_id and place_id
future_df = future_df.merge(products[['sku', 'item_id', 'place_id']], how='left', on=['sku'])

if add_remainders:
    future_df = create_aggr.bushe_strange_rule(future_df, expiry_rule)

##### Send forecasts to table
result_shape = save_results.send_forecast_to_plot(future_df, products, last_trans_date, day_after_today, forecast_url, forecast_plan_url, plans_url)


if smart_rounding:
    result_shape = save_results.send_forecast_to_table(future_df, last_trans_date, day_after_today, forecast_url, settings=settings)
else:
    result_shape = save_results.send_forecast_to_table(future_df, last_trans_date, day_after_today, forecast_url)

##### If possible save this data so we can have it for future reference. Can be useful.
#valid_grid['fcast_date'] = fcast_date
#valid_grid['valid_weeks'] = valid_weeks
#valid_grid['feats_weeks'] = feats_weeks
#valid_grid['model_name'] = model_name
#valid_grid['client'] = client
#valid_grid # append this df to a db

print(' model donee')