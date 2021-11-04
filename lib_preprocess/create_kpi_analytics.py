
import pandas as pd


# =============================================================================
# Создаем данные по kpi - создаем аналитическую таблицу
# =============================================================================
def create_analytics_dfs(df_func, additional_column = ['place_id'], time_column = ['date']):
    bill_count= df_func.groupby(time_column + additional_column)['external_id'].nunique().reset_index().rename(columns={'external_id':'value'})
    avg_bill = df_func.groupby(time_column + additional_column).agg({'external_id':'nunique', 'item_discount':'sum'}).reset_index()

    avg_bill['value'] = avg_bill['item_discount'].div(avg_bill['external_id'])
    avg_bill = avg_bill[['value'] + time_column + additional_column]
    return bill_count, avg_bill