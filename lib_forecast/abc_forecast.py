import pandas as pd
import datetime



#за 100 дней берем продажи сколько списывалось, тренд был ли
#препроцессинг - модель по децилям
#залив что зеленое
# =============================================================================
# АВС анализ для отбора топовых позиций
# =============================================================================
def categorize_items(df_func, additional_column = ['place_id'], time_column = ['week'] , prev_days = 100):

    last_monday = str(df_func['date'].max().isocalendar()[0]) + '-W' + str(df_func['date'].max().isocalendar()[1] - 1)
    df_func = df_func[df_func['date'] < datetime.datetime.strptime(last_monday + '-1', "%Y-W%W-%w").date()]
    df_func = df_func[df_func['date'] > (last_date - datetime.timedelta(days=prev_days))]
    df_func['week'] = df_func['date'].apply(lambda x: x.isocalendar()[1])
    df_func['cost_price_per'] =  100 * df_func['cost_price'].div(df_func['item_discount'])
    avg_bill = df_func.groupby(time_column + additional_column).agg({'item_discount':'sum'}).reset_index()
    revenue_by_items = df_func.groupby(['item_id'] + time_column + additional_column).agg({'item_discount':'sum', 'sales':'sum', 'expiry':'sum', 'cost_price':'median', 'cost_price_per':'median'}).reset_index()
    revenue_by_items = revenue_by_items.merge(avg_bill[['item_discount'] + additional_column + time_column].rename(columns={'item_discount':'revenue_all'}), how='left', on = time_column + additional_column)

    i = 1
    for i in range(min(4, period // 7 - 2)):
        revenue_by_items['week_' + str(i + 1) + 'ago'] = revenue_by_items.groupby(additional_column + ['item_id'])['item_discount'].shift(i+1)
    revenue_by_items = revenue_by_items.reset_index()
    #int_cols = revenue_by_items.columns[revenue_by_items.columns.str.startswith('week_')].values.tolist()

    #revenue_by_items['sales_' + str(i + 1) + 'w_trend'] = revenue_by_items.apply(lambda x:  - 100 * np.polyfit(np.arange(0, len(int_cols)), list(x[int_cols]), 1)[0] /x['week_' + str(i + 1) + 'ago']  if x['week_' + str(i + 1) + 'ago'] > 0 else 0, axis=1)

    int_cols_3 = ['week_1ago', 'week_2ago', 'week_3ago', 'week_4ago']

    revenue_by_items['sales_short_trend'] = revenue_by_items.apply(lambda x:  - 100 * np.polyfit(np.arange(0, len(int_cols_3)), list(x[int_cols_3]), 1)[0] /x['week_4ago']  if x['week_' + str(i + 1) + 'ago'] > 0 else 0, axis=1)

    revenue_by_items['var'] = revenue_by_items[int_cols_3].std(axis=1).div(revenue_by_items['week_1ago'])
    revenue_by_items['var'].fillna(1, inplace=True)
    # Исключаем текущую неделю из данных
    revenue_by_items = revenue_by_items[revenue_by_items.week + 1 > revenue_by_items.week.max()].groupby(['item_id'] + additional_column)[['item_discount', 'revenue_all', 'sales', 'var', 'sales_short_trend', 'expiry', 'cost_price', 'cost_price_per']].mean().reset_index()

    revenue_by_items['item_percent'] = 100 * revenue_by_items['item_discount'].div(revenue_by_items['revenue_all'])
    revenue_by_items = revenue_by_items[~(revenue_by_items['item_percent']==0)]

    up_lim_rev = revenue_by_items.groupby(additional_column)['item_percent'].quantile(0.8).reset_index().rename(columns={'item_percent':'up_lim_rev'})
    revenue_by_items = revenue_by_items.merge(up_lim_rev, on = additional_column, how='left')
    down_lim_rev = revenue_by_items.groupby(additional_column)['item_percent'].quantile(0.2).reset_index().rename(columns={'item_percent':'down_lim_rev'})
    revenue_by_items = revenue_by_items.merge(down_lim_rev, on = additional_column, how='left')
    up_lim_dyn = revenue_by_items.groupby(additional_column)['sales_short_trend'].quantile(0.8).reset_index().rename(columns={'sales_short_trend':'up_lim_dyn'})
    revenue_by_items = revenue_by_items.merge(up_lim_dyn, on = additional_column, how='left')
    down_lim_dyn = revenue_by_items.groupby(additional_column)['sales_short_trend'].quantile(0.2).reset_index().rename(columns={'sales_short_trend':'down_lim_dyn'})
    revenue_by_items = revenue_by_items.merge(down_lim_dyn, on = additional_column, how='left')
    up_lim_cost = df_func.groupby(additional_column)['cost_price'].quantile(0.8).reset_index().rename(columns={'cost_price':'up_lim_cost'})
    revenue_by_items = revenue_by_items.merge(up_lim_cost, on = additional_column, how='left')
    down_lim_cost = df_func.groupby(additional_column)['cost_price'].quantile(0.2).reset_index().rename(columns={'cost_price':'down_lim_cost'})
    revenue_by_items = revenue_by_items.merge(down_lim_cost, on = additional_column, how='left')
    up_lim_cost_per = df_func.groupby(additional_column)['cost_price_per'].quantile(0.8).reset_index().rename(columns={'cost_price_per':'up_lim_cost_per'})
    revenue_by_items = revenue_by_items.merge(up_lim_cost_per, on = additional_column, how='left')
    down_lim_cost_per = df_func.groupby(additional_column)['cost_price_per'].quantile(0.2).reset_index().rename(columns={'cost_price_per':'down_lim_cost_per'})
    revenue_by_items = revenue_by_items.merge(down_lim_cost_per, on = additional_column, how='left')

    revenue_by_items['category_rev'] = (np.where(revenue_by_items['item_percent'] > revenue_by_items['up_lim_rev'], 'top_share', np.where((revenue_by_items['item_percent'] < revenue_by_items['down_lim_rev']), 'down_share', 'typical_share')))

    revenue_by_items['category_dyn'] = np.where((revenue_by_items['sales_short_trend'] > revenue_by_items['up_lim_dyn']) &(revenue_by_items['var'] < 0.4), 'top_dynamics', np.where((revenue_by_items['sales_short_trend'] < revenue_by_items['down_lim_dyn']), 'down_dynamics', 'typical_dynamics'))
    #для себестоимости обратные соотношения!
    revenue_by_items['category_cost'] = (np.where(revenue_by_items['cost_price_per'] < revenue_by_items['down_lim_cost_per'], 'top_cost', np.where((revenue_by_items['cost_price_per'] > revenue_by_items['up_lim_cost_per']), 'down_cost', 'typical_cost')))

    revenue_by_items['value'] = revenue_by_items.apply(lambda x: 1 if (x['category_rev']=='top_share') |((x['category_dyn']=='top_dynamics')&(x['category_rev']!='down_share')) else 0, axis=1)

    revenue_by_items.loc[:,'value'] = revenue_by_items.apply(lambda x: 1 if (x['category_cost']=='top_cost') else x['value'], axis=1)

    revenue_by_items.loc[:,'value'] = revenue_by_items.apply(lambda x: x['value'] if ((x['sales']<10)&(x['expiry']<0.25*x['sales']))|((x['sales']>=10)&(x['sales']<30)&(x['expiry']<0.2*x['sales']))|((x['sales']>=30)&(x['expiry']<0.15*x['sales'])) else 0, axis=1)
    revenue_by_items = revenue_by_items[['value', 'item_id'] + additional_column]

    for place in revenue_by_items['place_id'].drop_duplicates():
        temp_url = forecast_green_light_url + place + '/showcase_highlights/upsert'
        df_new = revenue_by_items[revenue_by_items['place_id']==place][['place_id', 'item_id', 'value']].drop_duplicates().rename(columns={'value':'green'})
        data = df_new.to_dict('records')
        example_forecats = {'token': token, 'highlights': data}
        r = requests.post(url = temp_url, json = example_forecats)
    print(r.text)
    return revenue_by_items