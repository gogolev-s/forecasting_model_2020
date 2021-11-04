import pandas as pd
import datetime
import requests


# =============================================================================
# Загрузка транзакций через аккаунт
# =============================================================================
def load(items_url):
    items_req = requests.get(items_url).json()
    if 'items' in items_req:
        items_dict = items_req['items']
        items = pd.DataFrame.from_dict(items_dict)
        items['category'].replace({None:''}, inplace=True)
        items['name'].replace({None:''}, inplace=True)
        items_group = items[items['group']]
        return items, items_group
    else:
         print('No items!')
   
    
