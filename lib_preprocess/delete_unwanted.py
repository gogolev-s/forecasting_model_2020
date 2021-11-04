import pandas as pd
import pickle
import os
#df_func=transactions

# =============================================================================
# Удаляем не блюда
# =============================================================================
def remove_bad_products(df_func, bad_masks, bad_values, appropriate_values, feature):
    # все категории
    values = pd.DataFrame(df_func[feature].unique()).rename(columns={0:feature})
    values = pd.DataFrame(values.apply(lambda x: x[feature].lower(), axis=1)).rename(columns={0:feature})
    df_func.loc[:,feature] = df_func[feature].apply(lambda x: x.lower())
    # категории, которые мы убираем из рассмотрения
    for cat in bad_masks:
        bad_values.extend(values[values[feature].str.contains(cat)][feature].tolist())
    for cat in appropriate_values:
        if cat in bad_values: bad_values.remove(cat)
    df_func = df_func[~(df_func[feature].isin(bad_values))]

    return df_func

def remove_new_places_items(df_func, abs_path, last_date):
    columns_path = os.path.join(abs_path, "data", "models", os.getenv('account') + '_id.txt')
    if os.path.exists(columns_path):
        with open(columns_path, 'rb') as handle:
            all_dict = pickle.loads(handle.read())

        bad_places = df_func['place_id'].unique().tolist()
        bad_items = df_func['item_id'].unique().tolist()
        
        bad_places = list(set(bad_places) - set(all_dict['place_id']))
        bad_items = list(set(bad_items) - set(all_dict['item_id']))
        
        df_func = df_func[~df_func['place_id'].isin(bad_places)]
        df_func = df_func[~df_func['item_id'].isin(bad_items)]
        
    return df_func