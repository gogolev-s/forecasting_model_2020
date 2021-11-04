import pandas as pd
import pickle
import os
import datetime
from workalendar.europe import Russia

additional_holidays = [
    (datetime.date(2020, 1, 7), 'Новогодние праздники'),
    (datetime.date(2020, 1, 8), 'Новогодние праздники'),
    (datetime.date(2020, 3, 9), 'Международный женский день (перенос)'),
    (datetime.date(2020, 5, 4), 'Праздник Весны и Труда (перенос)'),
    (datetime.date(2020, 5, 5), 'Праздник Весны и Труда (перенос)'),
    (datetime.date(2020, 6, 24), 'Парад Победы'),
    (datetime.date(2020, 7, 1), 'День голосования за поправки в Конституцию'),
    (datetime.date(2020, 9, 1), 'День знаний'),
    (datetime.date(2021, 1, 7), 'Новогодние праздники'),
    (datetime.date(2021, 1, 8), 'Новогодние праздники'),
    (datetime.date(2021, 2, 23), 'День защитиника Отечества'),
    (datetime.date(2021, 3, 8), 'Международный женский день')]


def holidays_future(last_date):
    cal = Russia()
    years = set([last_date.year, (last_date + datetime.timedelta(days=16)).year])
    holidays = pd.DataFrame()
    for year in years:
        holidays = pd.concat([holidays, pd.DataFrame(cal.holidays(year))], sort=False)

    holidays = pd.concat([holidays, pd.DataFrame(additional_holidays)], sort=False)

    holidays_translate = {'New year':'Новый год',
                      'Day After New Year':'Новогодние праздники',
                      'Christmas':'Рождество',
                      'Defendence of the Fatherland':'День защитника Отечества',
                      "International Women's Day":'Международный женский день',
                      'Labour Day':'Праздник Весны и Труда',
                      'Victory Day':'День Победы',
                      'National Day':'День России',
                      'Day of Unity':'День народного единства'}
    holidays.rename(columns={0:'date', 1:'name'}, inplace=True)
    holidays['name'].replace(holidays_translate, inplace=True)

    pre_holidays = holidays.copy()
    pre_holidays['date'] = pre_holidays['date'] - datetime.timedelta(days=1)

    pre2_holidays = pre_holidays.copy()
    pre2_holidays['date'] = pre2_holidays['date'] - datetime.timedelta(days=1)

    pre_holidays = pre_holidays[pre_holidays['date'] > last_date]
    pre2_holidays = pre2_holidays[pre2_holidays['date'] > last_date]
    holidays = holidays[holidays['date'] > last_date]


    pre_holidays = pre_holidays[pre_holidays['date'] < last_date + datetime.timedelta(days=16)]
    pre2_holidays = pre2_holidays[pre2_holidays['date'] < last_date + datetime.timedelta(days=16)]
    holidays = holidays[holidays['date'] < last_date + datetime.timedelta(days=16)]
    return pre2_holidays, pre_holidays, holidays

def holidays(last_date, training_weekday, abs_path, holidays_period=400, items_group = pd.DataFrame()):
    cal = Russia()
    first_date_holidays = last_date - datetime.timedelta(days=holidays_period)
    years = set([first_date_holidays.year, last_date.year, (last_date + datetime.timedelta(days=10)).year])
    holidays=[]
    for year in years:
        hol = cal.holidays(year)
        for i in range(len(hol)):
            hol[i] = hol[i][0]
        holidays.extend(hol)
    for hol in additional_holidays:
        holidays.append(hol[0])

    holidays_before = [i - datetime.timedelta(days=1) for i in holidays]
    holidays_after = [i + datetime.timedelta(days=1) for i in holidays]

    need_for_holidays = False
    for i in range(1, 10):
        if last_date + datetime.timedelta(days=i) in (holidays_before):
            need_for_holidays=True

    holidays_dict = {'holiday':holidays, 'holiday_1after':holidays_after,'holiday_1before':holidays_before}

    if need_for_holidays:
        model_path = os.path.join(abs_path, "data", "models", os.getenv('account') + '_holiday.txt')
    else:
        model_path = os.path.join(abs_path, "data", "models", os.getenv('account') + '.txt')

    groups_path = os.path.join(abs_path, "data", "transactions", os.getenv('account') + '_groups.txt')

    if os.path.exists(groups_path):
        with open(groups_path, 'rb') as handle:
            old_groups = pickle.loads(handle.read())
    else:
        old_groups = 'not_df'

    if items_group.shape[0] > 0:
        with open(groups_path, 'wb') as handle:
            pickle.dump(items_group, handle)

    #Special model, training weekday, KPI sending
    train_model_today = (not os.path.exists(model_path)) | (not items_group.equals(old_groups)) | (last_date.weekday() == training_weekday) | (last_date.day <= 2)
    return holidays_dict, need_for_holidays, train_model_today


def add_holidays_factors(df_func, holidays_dict):

    df_func['holiday'], df_func['holiday_1before'], df_func['holiday_1after'] = 0, 0, 0

    for key in holidays_dict:
        dates = holidays_dict[key]
        df_func.loc[df_func['date'].isin(dates), key] = 1

    return df_func
