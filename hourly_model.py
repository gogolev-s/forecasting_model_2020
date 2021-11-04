# -*- coding: utf-8 -*-
import os
import sys
import datetime
import pickle
import pandas as pd
import lightgbm as lgb
from dotenv import load_dotenv

if os.name == 'nt':
    #abs_path = 'Z:/foodcast/forecast_meta_models_test/src'
    abs_path = 'E:/Git/forecast_meta_models_test/src'
    os.chdir(abs_path)
else:
    abs_path = os.path.dirname(os.path.abspath(__file__))

load_dotenv(dotenv_path=os.path.join(abs_path, "data", 'info_dev.env'))

##### Set variables
base_url = os.getenv('ML_API_BASE_URL')
token = os.getenv('ML_API_KEY')
##### API token
token_url = 'token=' + token

pd.options.mode.chained_assignment = None

from lib_load_internal import places as load_places
from lib_load_internal import items as load_items
from lib_load_internal import transactions as load_transactions
from lib_preprocess import create_aggr
from lib_forecast import kissgbm as kgbm
from lib_save import save_results_hour


##### Load info about all clients
with open(os.path.join(abs_path, "clients_info.txt"), 'rb') as handle:
    client_params = pickle.loads(handle.read())

client = sys.argv[1]

#client = 'Korzhov' #### For testing
client_pars = client_params[client]

expiry_reason = []
include_today = False


account = client_pars['account']
dummies = client_pars['dummies']
training_weekday = client_pars['training_weekday']
first_transaction = client_pars['first_transaction']
rule_expiry, add_remainders = False, False

if 'expiry_reason' in client_pars: expiry_reason = client_pars['expiry_reason']
if 'day_after_today' in client_pars: day_after_today = client_pars['day_after_today']
if 'rule_expiry' in client_pars: rule_expiry = client_pars['rule_expiry']
if 'add_remainders' in client_pars: add_remainders = client_pars['add_remainders']
#account='d8d013a6-5b48-482e-bbb8-91eaaf22ebc5'
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
event_weeks = 13 # model uses maximum 18 weeks of events to create models
feats_weeks = 3 # each event uses another 5 weeks for feature engineering -- note, this does not impact f.e. automatically as of now, need to add that functionality later
period = 7 * valid_weeks + 7 * event_weeks + 7 * feats_weeks
predict_days = 6 # how many days ahead to predict, t+1, t+2, etc..
day_after_today = 1

##### The last date with transactions, + determine fcast_date
last_trans_date = datetime.date.today() - datetime.timedelta(days=1 - include_today)
fcast_date = last_trans_date + datetime.timedelta(days=1)

##### Threshold date for data loading
first_trans_date = last_trans_date - datetime.timedelta(days=period)
##### Date of first ransaction by the client in order to avoid empty requests
first_trans_date = max(first_trans_date, first_transaction)

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

##### Keep only retail sales
if min_retail_amount > 0: transactions = transactions[transactions['sales'] <= min_retail_amount]

##### Calculate actual price = item_discount
transactions.loc[:, 'item_discount'] = transactions['item_discount'] / (transactions['amount'])
transactions.drop(['external_id', 'amount'], axis=1, inplace=True)

##### Transformation some groups if they exist
if items_group.shape[0] > 0:
    transactions = create_aggr.transform_groups(transactions, base_url, items_group)

##### Calculate how much is sold at each hour to recaluate sales in days without expiries
hours_dict = create_aggr.sales_by_hours(transactions, rule_expiry)

##### Transform transactions to daily df
df = create_aggr.create_df_hour(transactions)
df.sales.fillna(0, inplace=True)

# =============================================================================
# Ivan's old code
# =============================================================================

df['sku'] = df['place_id'] + df['item_id'] # sku unique id feature

##### Keep table with cost and prices and combination of sku and item/place id in order to get it back in the end
products = create_aggr.create_nomenclature(df, with_expiry, last_trans_date)

# =============================================================================
# kissgbm create best model
# =============================================================================

# save original df for later use when forecasting and keep only used features with proper name
df.drop_duplicates(['date', 'hour', 'sku'], inplace=True)
df_ori = df.copy()
df = df[['date', 'hour', 'sales', 'sku']].rename(columns={'sales':'y'})

# Drop columns that almost don't vary
df.drop(df.std()[df.std()/df.mean() < 0.1].index.values, axis=1, inplace=True)
# create features kgbm way
fe_cutoff_date = fcast_date - datetime.timedelta(days=7 * valid_weeks + 7 * event_weeks) # set the first date of data to be used after f.e. is done
df = kgbm.create_features(df, fe_cutoff_date, hourly=True) # create features with cutoff date selection

# set validation dates
valid_last = fcast_date - datetime.timedelta(days=1)
valid_first = fcast_date - datetime.timedelta(days=7 * valid_weeks)

# run booster grid
params_list = kgbm.params_list() # parameters to evaluate
valid_grid = kgbm.valid_grid(df, params_list, valid_last, valid_first, hourly=True)

# select best booster found according to validation grid
best_valid = valid_grid.sort_values('swaa_va', ascending=False).iloc[0]
best_booster = lgb.Booster(model_str=best_valid['booster_string'], silent=True)

for step in range(predict_days):
    step_fcast_date = fcast_date + datetime.timedelta(days=step)

    # grab skus that were present the past weekday and append past data for them only
    mask = df_ori['date'] == step_fcast_date - datetime.timedelta(days=7)
    fc_df = df_ori[mask][['hour', 'sku']].copy()
    fc_df['date'] = step_fcast_date # create forecast date column
    sku_list = fc_df['sku'].unique() # get list of skus to forecast
    fc_df = fc_df.append(df_ori[df_ori['sku'].isin(sku_list)], sort='True')

    # f.e.
    fc_df = fc_df[['date', 'sales', 'sku', 'hour']].rename(columns={'sales':'y'})
    fc_df = kgbm.create_features(fc_df, step_fcast_date, hourly=True).drop(columns='y') # now this cutoff will always yield only 1 date, the one we are forecasting

    # predict and round
    fc_df['y'] = best_booster.predict(fc_df[best_booster.feature_name()])
    fc_df['y'] = round(fc_df['y'])

    # save forecasts to original for next iteration
    df_ori = df_ori.append(fc_df[['date', 'sku', 'hour', 'y']].rename(columns={'y':'sales'}), sort='True')

# select forecasts
future_df = df_ori[df_ori['date'] >= fcast_date][['date', 'sku', 'hour', 'sales']].copy()

# merge product data, this now only gets item_id and place_id
future_df = future_df.merge(products[['sku', 'item_id', 'place_id']], how='left', on=['sku'])

##### Send forecasts to table

tz=places['tz'].iloc[0]

result_shape = save_results_hour.to_plot(future_df, products, last_trans_date, day_after_today, forecast_url, tz)

result_shape = save_results_hour.to_table(future_df, last_trans_date, day_after_today, forecast_url, tz)

print(' model donee')