# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import lightgbm as lgb
import itertools
import datetime

def create_features(df, cutoff_date, hourly=False):
    '''
    - This will take df and create aggregations with simple merges.
    - To remove data that was only there to create features, first_date is used.
    '''

    sku = ['sku'] + hourly * ['hour']
    df = df.copy() # this is to create function-linked df, otherwise global df is affected
    # get yXago columns
    for day in range(1, 3 * 7 + 1):
        day_field = str(day) + 'ago'
        df[day_field] = df['date'] + datetime.timedelta(days = day)
        df = df.merge(df[[day_field,'y'] + sku], left_on = ['date'] + sku, right_on = [day_field] + sku, how = 'left')
        df = df.rename(columns = {'y_x':'y','y_y':'y' + day_field})
        df = df.drop(columns = [day_field + '_x',day_field + '_y'])

    # remove history data, we don't need it anymore
    df = df[df['date'] >= cutoff_date]
    # remove skus that when trying to predict were not present 7 days ago, because this is the logic to use when forecasting
    if hourly:
        df = df[df['y7ago'].notnull() | df['y14ago'].notnull() | df['y21ago'].notnull()]
    else:
        df = df[df['y7ago'].notnull()]
    # wdays aggregations
    wdays_list = []
    for day in range(7, 3 * 7 + 1,7):
        day = str(day)
        current_day = 'y' + day + 'ago'
        wdays_list.append(current_day)
        df['y_wd_' + day] = df[current_day]
        if len(wdays_list) > 1: # skip the first wd, because it consists of only 1 value
            for agg in ['mean','median','min','max','std','count']:
                df['y_wd_' + day + '_' + agg] = df[wdays_list].agg(agg,axis = 1)

    # week aggregations, stopping based on wdays_list
    days_list = []
    for day in range(1, 3 * 7 + 1,1):
        day = str(day)
        current_day = 'y' + day + 'ago'
        days_list.append(current_day)
        if current_day in wdays_list: # in this way we are sure to get aggs for week intervals
            for agg in ['mean','median','min','max','std','count']:
                df['y_wk_' + day + '_' + agg] = df[days_list].agg(agg,axis = 1)

    # set 2 = rat and dif of important features as seen by split/gain importance
    df['s2_y1ago_y_wk_7_mean_rat'] = df['y1ago'] / df['y_wk_7_mean']
    df['s2_y1ago_y_wk_7_mean_dif'] = df['y1ago'] - df['y_wk_7_mean']
    df['s2_y1ago_y_7ago_rat'] = df['y1ago'] / df['y7ago']
    df['s2_y1ago_y_7ago_dif'] = df['y1ago'] - df['y7ago']
    df['s2_7ago_wd14_mean_rat'] = df['y7ago'] / df['y_wd_14_mean']
    df['s2_7ago_wd14_mean_dif'] = df['y7ago'] - df['y_wd_14_mean']
    df['s2_7ago_14ago_rat'] = df['y7ago'] / df['y14ago']
    df['s2_7ago_14ago_dif'] = df['y7ago'] - df['y14ago']
    df['s2_6ago_13ago_rat'] = df['y6ago'] / df['y13ago']
    df['s2_6ago_13ago_dif'] = df['y6ago'] - df['y13ago']
    df['s2_1ago_8ago_rat'] = df['y8ago'] / df['y8ago']
    df['s2_1ago_8ago_dif'] = df['y8ago'] - df['y8ago']
    df['s2_wk7_mean_wk14_mean_rat'] = df['y_wk_7_mean'] / df['y_wk_14_mean']
    df['s2_wk7_mean_wk14_mean_dif'] = df['y_wk_7_mean'] - df['y_wk_14_mean']
    df['s2_wk7_median_wk14_median_dif'] = df['y_wk_7_median'] / df['y_wk_14_median']
    df['s2_wk7_median_wk14_median_dif'] = df['y_wk_7_median'] - df['y_wk_14_median']
    df['s2_wk7_max_wk14_min_rat'] = df['y_wk_7_max'] / df['y_wk_7_min']
    df['s2_wk7_max_wk14_min_dif'] = df['y_wk_7_max'] - df['y_wk_7_min']
    df['s2_wd14_max_wd14_min_rat'] = df['y_wd_14_max'] / df['y_wd_14_min']
    df['s2_wd14_max_wd14_min_dif'] = df['y_wd_14_max'] - df['y_wd_14_min']

    # set 3 = yury insight on salary days, pass on continuous month day attibute, lightgbm should find best splits
    df['s3_monthday'] = pd.to_datetime(df['date']).dt.day

    # weekday feature
    df['weekday'] = pd.Categorical(pd.to_datetime(df['date']).dt.dayofweek)

    return df

# custom swaa metric for early stopping
def swaa(preds,train_data):
    labels = train_data.get_label()
    preds = np.round_(preds) # rounding that will be done on prediction also
    abs_errors = abs(preds - labels)
    swaa = 1 - abs_errors.sum() / labels.sum()
    eval_name = 'swaa'
    eval_result = swaa
    is_higher_better = True
    return eval_name,eval_result,is_higher_better

# different boosters to evaluate
def params_list():
    vec_objective = ['poisson','tweedie','regression'] # quantile or mape were not performing well
    vec_leaves = [8]
    vec_rate = [0.02]
    vec_fe_frac = [0.25]
    vec_event_weeks = [3,6,9,12,15,18]
    params_list = []
    for objective,leaves,rate,fe_frac,event_weeks in itertools.product(
        vec_objective,
        vec_leaves,
        vec_rate,
        vec_fe_frac,
        vec_event_weeks
    ):
        lgbm = {
            'metric':'None', # because we use custom swaa metric
            'objective':objective,
            'num_leaves':leaves,
            'learning_rate':rate,
            'feature_fraction':fe_frac,
            'verbosity':-1,
            'random_state':7,
        }
        proc = {
            'event_weeks':event_weeks,
        }
        params_list.append([lgbm,proc])
    return params_list

def valid_grid(df,params_list, valid_last, valid_first, hourly=False):
    df = df.copy() # this is to create function-linked df, otherwise global df is affected
    sku = ['sku'] + hourly * ['hour']

    # set fixed validation dataframe
    va_df = df[(df['date'] <= valid_last) & (df['date'] >= valid_first)].copy()

    # train_last date is always 1 day before_valid first, train_first will be defined on each iteration based on event_weeks
    train_last = valid_first - datetime.timedelta(days = 1)

    # repository for each parameter iteration
    valid_grid = pd.DataFrame()
    for params in params_list:
        # unwrap lgbm and procedure parameters dictionaries
        lgbm = params[0]
        proc = params[1]

        # set train_first based on current event_weeks
        event_weeks = proc['event_weeks']
        train_first = valid_first - datetime.timedelta(days = event_weeks * 7)

        # train df lgb.dataset, valid lgb.dataset
        tr_df = df[(df['date'] <= train_last) & (df['date'] >= train_first)].copy()
        tr_ds = lgb.Dataset(tr_df.drop(columns = ['date', 'y'] + sku),label = tr_df['y'])
        va_ds = tr_ds.create_valid(va_df.drop(columns = ['date', 'y'] + sku),label = va_df['y'])

        # train booster
        val_res = {}
        booster = lgb.train(
            params = lgbm,
            train_set = tr_ds,
            num_boost_round = 10000,
            valid_sets = [tr_ds,va_ds],
            early_stopping_rounds = 20,
            feval = swaa,
            evals_result = val_res,
            verbose_eval = False,
        )

        # get results
        rounds = len(val_res['valid_1']['swaa'])
        swaa_tr = val_res['training']['swaa'][-1]
        swaa_va = val_res['valid_1']['swaa'][-1]

        # save current iteration results
        valid_grid = valid_grid.append(pd.DataFrame(data = {
            'event_weeks':event_weeks,
            'objective':lgbm['objective'],
            'num_leaves':lgbm['num_leaves'],
            'learning_rate':lgbm['learning_rate'],
            'feature_fraction':lgbm['feature_fraction'],
            'rounds':rounds,
            'swaa_tr':swaa_tr,
            'swaa_va':swaa_va,
            'booster_string':booster.model_to_string(),
        },index = [0]))
    return valid_grid