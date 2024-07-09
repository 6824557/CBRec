import pandas as pd
import time
import argparse
import numpy as np
import os
from geopy.distance import geodesic
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('--oriDataset',  default='GowallaCA')
# parser.add_argument('--sort_with_time',  default=0.2, type=float )
args = parser.parse_args()
dataset = ['NYC', 'TKY', 'SG', 'GB', 'GowallaCA']
rate = [0.8,0.6,0.4,0.2]


for rateNow in rate:
    for data in dataset:
        print(str(rateNow)+data)
        column_names = ['uid', 'pid', 'cid', 'venueCategory', 'latitude', 'longitude', 'Timestamp', 'date', 'week',
                        'month', 'zone']
        # train_with_time稀疏
        saveDir = data + "Glo"
        df = pd.read_csv('./'+saveDir+'/train_with_time.txt', sep='\t', header=None, names=column_names)
        sampled_df = df.sample(frac=rateNow, random_state=42)  # 设定随机种子以确保可重复性
        user = sampled_df['uid'].unique()
        user_to_id = dict(zip(list(user), list(np.arange(1, user.shape[0] + 1))))
        sampled_df['userid'] = sampled_df['uid'].map(user_to_id)
        sampled_df = sampled_df.sort_values(by=['userid','date'], ignore_index=True)
        sampled_df = sampled_df[
            ['userid', 'pid', 'cid', 'venueCategory', 'latitude', 'longitude', 'Timestamp', 'date', 'week', 'month',
             'zone']]
        sampled_file_path = f'./{saveDir}/train_with_time_{int(rateNow * 100)}.txt'
        sampled_df.to_csv(sampled_file_path, sep='\t', index=False,header=False)
