import pandas as pd
import numpy as np

def data_upload(path, normal, attack):
    '''Uploading files with normal and mixed system conditions '''
    normal_data = pd.read_excel(path + normal, header = 1)
    attack_data = pd.read_excel(path + attack, header = 1)
    return normal_data, attack_data

def data_preprocessing(normal_uploaded, attack_uploaded):
    '''Returns preproceed dataset with all existing data'''
    # Converting timestamp to date_time format
    normal_uploaded[' Timestamp'] = pd.to_datetime(normal_uploaded[' Timestamp'])
    attack_uploaded[' Timestamp'] = pd.to_datetime(attack_uploaded[' Timestamp'])
    # Removing gaps
    normal_uploaded.columns = normal_uploaded.columns.str.replace(' ', '')
    attack_uploaded.columns = attack_uploaded.columns.str.replace(' ', '')
    #Concating two datasets
    dataset = pd.concat([normal_uploaded, attack_uploaded], ignore_index=True)
    #Replacing dirty values
    dataset.loc[dataset['Normal/Attack'] == 'A ttack', 'Normal/Attack'] = 'Attack'
    dataset.loc[dataset['Normal/Attack'] == 'Attack', 'Normal/Attack'] = 1
    dataset.loc[dataset['Normal/Attack'] == 'Normal', 'Normal/Attack'] = 0
    return dataset

