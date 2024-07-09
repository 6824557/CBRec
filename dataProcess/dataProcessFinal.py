import pandas as pd
import time
import argparse
import numpy as np
import os
from geopy.distance import geodesic
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('--oriDataset',  default='NYC')
parser.add_argument('--sort_with_time',  default=1, type=int )
parser.add_argument('--dis',  default=5, type=int )
parser.add_argument('--filter',  default=1, type=int )
args = parser.parse_args()
if args.filter == 1:
    saveDir = args.oriDataset+"Glo"
if args.filter == 0:
    saveDir = args.oriDataset+"nofilter"
print(args.oriDataset)
print(saveDir)
data_pd = pd.read_csv(args.oriDataset+".csv",header=0,sep=',')
def filter_g_k_one(data,k=10,u_name='user_id',i_name='business_id',y_name='stars'):#筛选出记录中商品、用户次数大于k
    item_group = data.groupby(i_name).agg({y_name:'count'})
    item_g10 = item_group[item_group[y_name]>=k].index#找出交互中商品被交互大于10次
    data_new = data[data[i_name].isin(item_g10)]
    user_group = data_new.groupby(u_name).agg({y_name:'count'})
    user_g10 = user_group[user_group[y_name]>=k].index#找出用户交互数大于10次
    data_new = data_new[data_new[u_name].isin(user_g10)]
    return data_new

def filter_tot(data,k=10,u_name='user_id',i_name='business_id',y_name='stars'):
    data_new=data
    turn_num=0
    while True:
        data_new = filter_g_k_one(data_new,k=k,u_name=u_name,i_name=i_name,y_name=y_name)
        m1 = data_new.groupby(i_name).agg({y_name:'count'})
        m2 = data_new.groupby(u_name).agg({y_name:'count'})
        num1 = m1[y_name].min()
        num2 = m2[y_name].min()
        turn_num = turn_num + 1;
        print('item min:',num1,'user min:',num2)
        if num1>=k and num2>=k:
            break
        if turn_num >= 15:
            break
    return data_new
def calDis(data):
    # 创建一个字典用于存储地点之间的距离
    zone = {}
    data_num = data['venueId'].unique().shape[0]
    for i in range(data_num):
        zone[i+1]=0
    # 获取地点的经纬度坐标
    locations = data[['pid', 'latitude', 'longitude']].drop_duplicates(subset='pid')
    # 遍历所有地点的组合来计算它们之间的距离
    for i in locations.index:
        for j in locations.index:
            if i != j:
                location1 = (locations['latitude'][i], locations['longitude'][i])
                location2 = (locations['latitude'][j], locations['longitude'][j])
                distance = geodesic(location1, location2).kilometers
                if distance <= args.dis:
                    if zone[locations['pid'][i]]==0 and zone[locations['pid'][j]]==0:
                        zone[locations['pid'][i]]=locations['pid'][i]
                        zone[locations['pid'][j]] = locations['pid'][i]
                    elif zone[locations['pid'][i]]==0 and zone[locations['pid'][j]]!=0:
                        zone[locations['pid'][i]] = zone[locations['pid'][j]]
                    elif zone[locations['pid'][i]] != 0 and zone[locations['pid'][j]] == 0:
                        zone[locations['pid'][j]] = zone[locations['pid'][i]]
    data['zone'] = data['pid'].apply(lambda x: zone[x])
    return data

def normalize_frequency(frequency_data):
    freq_values = frequency_data['frequency']
    normalized_freq = (freq_values - freq_values.min()) / (freq_values.max() - freq_values.min())
    frequency_data['frequency'] = normalized_freq
    return frequency_data

if args.oriDataset in ['NYC','TKY']:
    data_pd = data_pd[['userId', 'venueId', 'venueCategoryId','venueCategory', 'latitude', 'longitude', 'utcTimestamp', 'timezoneOffset']]
elif args.oriDataset in ['GowallaCA']:
    data_pd = data_pd[['userId', 'venueId', 'venueCategoryId', 'venueCategory', 'latitude', 'longitude', 'utcTimestamp','timezoneOffset']]
else:
    data_pd = data_pd[['userId', 'venueId', 'utcTimestamp', 'timezoneOffset', 'latitude', 'longitude', 'venueCategory', 'country']]
    data_pd['venueCategoryId'] = data_pd['venueCategory']

data_pd['date'] = pd.to_datetime(data_pd['utcTimestamp'])
data_pd['month'] = data_pd['date'].dt.month
data_pd['week'] = data_pd['date'].dt.dayofweek
data_pd['Timestamp'] = data_pd['utcTimestamp'].apply(lambda x:time.mktime(time.strptime(x,'%a %b %d %H:%M:%S %z %Y')))
data_pd = filter_tot(data_pd, k=10, u_name='userId', i_name='venueId', y_name='userId')
#utils


#映射
user = data_pd['userId'].unique()
item = data_pd['venueId'].unique()
cate = data_pd['venueCategoryId'].unique()
#id映射为1到全部
user_to_id = dict(zip(list(user),list(np.arange(1, user.shape[0] + 1))))
item_to_id = dict(zip(list(item),list(np.arange(1, item.shape[0] + 1))))
cate_to_id = dict(zip(list(cate),list(np.arange(1, cate.shape[0] + 1))))
item_num = item.shape[0]
cate_num = cate.shape[0]
#
print("user num:",user.shape)
print("item num:", item.shape)
print("cate num:", cate.shape)
data_pd['uid'] = data_pd['userId'].map(user_to_id)
#将uid映射到所给字典中
data_pd['pid'] = data_pd['venueId'].map(item_to_id)
data_pd['cid'] = data_pd['venueCategoryId'].map(cate_to_id)
#增加区域列
data_pd = calDis(data_pd)
#排序
data_pd = data_pd.sort_values(by=['uid', 'date'], ignore_index=True)
data_train = data_pd[['uid','pid','cid','venueCategory','latitude','longitude','Timestamp','date', 'week', 'month','zone']]

path_folder = saveDir
if not os.path.exists("./"+path_folder):
    os.mkdir("./"+path_folder)
data_train.to_csv("./"+path_folder+"/train_with_time.txt",index=False,header=False,sep='\t')

total_visit = len(data_train)
weekend_visit = len(data_train[data_train['week'].isin([5, 6])])

#space
zone_user_counts = data_train.groupby('zone')['pid'].count().reset_index()
zone_user_counts.columns = ['zone', 'count']    #time
zone_user_counts['frequency'] = zone_user_counts['count'] / (total_visit+item_num)
data_train = data_train.merge(zone_user_counts[['zone', 'frequency']], on='zone', how='left')
pid_frequency = data_train[['pid', 'frequency']].drop_duplicates()
pid_frequency = normalize_frequency(pid_frequency)
#week
weekend_place_user_counts = data_train[data_train['week'].isin([5, 6])].groupby('pid')['uid'].count().reset_index()
weekend_place_user_counts.columns = ['pid', 'count']
weekend_place_user_counts['frequency'] = weekend_place_user_counts['count'] / (weekend_visit+item_num)
week_frequency = weekend_place_user_counts[['pid', 'frequency']].drop_duplicates()
week_frequency = normalize_frequency(week_frequency)
#point
place_user_counts = data_train.groupby('pid')['uid'].count().reset_index()
place_user_counts.columns = ['pid', 'count']
place_user_counts['frequency'] = place_user_counts['count'] / (total_visit+item_num)
point_frequency = place_user_counts[['pid', 'frequency']].drop_duplicates()
point_frequency = normalize_frequency(point_frequency)
#cate
# 计算每个类型被多少用户访问过
category_user_counts = data_train.groupby('cid')['uid'].count().reset_index()
category_user_counts.columns = ['cid', 'count']
category_user_counts['frequency'] = category_user_counts['count'] / (total_visit + cate_num)
cate_frequency = category_user_counts[['cid', 'frequency']].drop_duplicates()
cate_frequency = normalize_frequency(cate_frequency)



filename = f'{saveDir}/space_pop_seq_ori.txt.txt'
pid_frequency.to_csv(filename, sep='\t', index=False)  # 使用制表符分隔，并且不保存行索引
filename = f'{saveDir}/time_pop_seq_ori.txt'
week_frequency.to_csv(filename, sep='\t', index=False)  # 使用制表符分隔，并且不保存行索引
filename = f'{saveDir}/item_pop_seq_ori.txt'
point_frequency.to_csv(filename, sep='\t', index=False)  # 使用制表符分隔，并且不保存行索引
filename = f'{saveDir}/cate_pop_seq_ori.txt.txt'
cate_frequency.to_csv(filename, sep='\t', index=False)  # 使用制表符分隔，并且不保存行索引
print(data_train.head())

