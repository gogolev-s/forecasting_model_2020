import pandas as pd
import requests


def load(places_url):
    places =  requests.get(places_url).json()
    if 'places' in  places:
        places = pd.DataFrame.from_dict(places['places'])
        places.rename(columns={'id':'place_id'}, inplace=True)
    else:
        print('No places!')
    return places
