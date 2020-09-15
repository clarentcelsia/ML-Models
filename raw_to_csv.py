# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:31:12 2020

@author: user
"""

# Import file from github and write it into csv

import requests as rq


url = "https://raw.githubusercontent.com/clarentclaire/DataClassification/master/temp_dataset/diabetes.csv"

downloaded_filename = "diabetes.csv"

def raw_to_csv():
    print('Downloading...')

    response = rq.get(url)
    response.raise_for_status()
    with open(downloaded_filename, 'wb') as f:
        f.write(response.content)
    
    print('Download succeed!')


if __name__ == "__main__":
    raw_to_csv()