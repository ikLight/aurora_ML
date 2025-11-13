import json
import pandas as pd
import requests as re


##-----------------------------------------------------------------------------##

class Data():

    def __init__(self):
        pass

    #--------------------------------------#

    def _convert_to_csv(self, data):
        return pd.concat([pd.DataFrame([item]) for item in data["items"]], ignore_index=True)
    
    #--------------------------------------#
    
    def get_data(self, url):
        response = re.get(url)
        if response.status_code == 200:
            data = self._convert_to_csv(response.json())  
            return data            
        else:
            print(f"Request failed with status {response.status_code}: {response.text}")

##-----------------------------------------------------------------------------##


## test

# API_ROUTE = "https://november7-730026606190.europe-west1.run.app/messages/?skip=0&limit=4000"
# test = Data().get_data(API_ROUTE)
# print(test.shape)
# test.to_csv('data.csv', index=False)

