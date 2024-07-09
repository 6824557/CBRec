import pandas as pd
import time
import argparse
import numpy as np
import os
from geopy.distance import geodesic
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('--oriDataset',  default='NYC')
parser.add_argument('--saveDir',  default='NYCProGlo')
parser.add_argument('--slot_count',  default=10 )
parser.add_argument('--filter',  default=10 ,type=int)
parser.add_argument('--sort_with_time',  default=1, type=int )
args = parser.parse_args()
print(args.oriDataset)
print(args.saveDir)
data_pd = pd.read_csv(args.oriDataset+".csv",header=0,sep=',')
data_pd = data_pd[['userId','venueId','venueCategoryId','latitude','longitude','utcTimestamp','timezoneOffset']]
data_pd['date'] = pd.to_datetime(data_pd['utcTimestamp'])
data_pd['year'] = data_pd['date'].dt.year
data_pd['month'] = data_pd['date'].dt.month
data_pd.head()

#utils
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
    while True:
        data_new = filter_g_k_one(data_new,k=k,u_name=u_name,i_name=i_name,y_name=y_name)
        m1 = data_new.groupby(i_name).agg({y_name:'count'})
        m2 = data_new.groupby(u_name).agg({y_name:'count'})
        num1 = m1[y_name].min()
        num2 = m2[y_name].min()
        print('item min:',num1,'user min:',num2)
        if num1>=k and num2>=k:
            break
    return data_new

def calDis(data):
    # 创建一个字典用于存储地点之间的距离
    dis = {}
    data_pd = data
    # 获取地点的经纬度坐标
    locations = data_pd[['pid', 'latitude', 'longitude']].drop_duplicates(subset='pid')
    # 遍历所有地点的组合来计算它们之间的距离
    for i in locations.index:
        for j in locations.index:
            if i != j:
                location1 = (locations['latitude'][i], locations['longitude'][i])
                location2 = (locations['latitude'][j], locations['longitude'][j])
                distance = geodesic(location1, location2).kilometers
                dis[locations['pid'][i]] = dis.get(locations['pid'][i], {})
                dis[locations['pid'][i]][locations['pid'][j]] = distance

    return dis
data = filter_tot(data_pd, k=args.filter, u_name='userId', i_name='venueId', y_name='userId')

data['Timestamp'] = data['utcTimestamp'].apply(lambda x:time.mktime(time.strptime(x,'%a %b %d %H:%M:%S %z %Y')))
time_min = data['Timestamp'].min()
time_max = data['Timestamp'].max()
slot_gap = (time_max - time_min) /10#分的时候是按照时间片分的，每个时间片可能有许多相同时间的交互，所以每个时间片的交互数量不同
data['time_slot'] = data["Timestamp"].apply(lambda x: int(min(int((x-time_min))//slot_gap,9)))
data['time_slot'] = data[['time_slot']].astype(np.int32)
timestamp = time_min + slot_gap
data.head()

#转换成localtime,不要测试集了
time_local = time.localtime(timestamp)
dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)

train_slots = [0 ,1, 2, 3, 4, 5, 6,7,8,9]
data_train = data[data['time_slot'].isin(train_slots)]

user_in_train = data_train['userId'].copy()#unique去掉重复的元素,不需要去掉，地点
item_in_train = data_train['venueId'].copy()

#映射
user = data_train['userId'].unique()
item = data_train['venueId'].unique()
cate = data_train['venueCategoryId'].unique()
#id映射为0到全部
user_to_id = dict(zip(list(user),list(np.arange(1, user.shape[0] + 1))))
item_to_id = dict(zip(list(item),list(np.arange(1, item.shape[0] + 1))))
cate_to_id = dict(zip(list(cate),list(np.arange(1, cate.shape[0] + 1))))
#
print("user num:",user.shape)
print("item num:", item.shape)
print("cate num:", cate.shape)
data_train['uid'] = data_train['userId'].map(user_to_id)
#将uid映射到所给字典中
data_train['pid'] = data_train['venueId'].map(item_to_id)
data_train['cid'] = data_train['venueCategoryId'].map(cate_to_id)

#我有重复数据，必须排序
if args.sort_with_time == 1:
    data_train = data_train.sort_values(by=['uid', 'date'], ignore_index=True)
else:
    data_train = data_train.sort_values(by=['uid'], ignore_index=True)

data_train = data_train[['uid','pid','cid','latitude','longitude','time_slot']]

columns = ['uid','pid','cid','latitude','longitude','time_slot']
data_train.columns = columns
data_train.head()


#save
path_folder = args.saveDir
print(path_folder)
if not os.path.exists("./"+path_folder):
    os.mkdir("./"+path_folder)
data_train.to_csv("./"+path_folder+"/train_with_time.txt",index=False,header=False,sep=' ')

for slot_id in train_slots:
    slot_data = data_train[data_train['time_slot'].isin([slot_id])]
    slot_data  = slot_data.sort_values(by=['pid'],ignore_index=True)
    slot_data_np = slot_data[['pid','uid']].values[:,0:2]

    with open("./"+path_folder+"/cr_"+str(slot_id)+".txt",'w') as f:
        i_pre = slot_data_np[0,0]
        k = 0
        for x in slot_data_np:
            i_ = x[0]
            u_ = x[1]
            if i_ != i_pre or k == 0:
                i_pre = i_
                if k>0:
                    f.write('\n')
                f.write(str(i_))
                k = 1
            f.write(" " + str(u_))

for slot_id in train_slots:
    slot_data = data_train[data_train['time_slot'].isin([slot_id])]
    slot_data  = slot_data.sort_values(by=['cid'],ignore_index=True)
    slot_data_np = slot_data[['cid','uid']].values[:,0:2]

    with open("./"+path_folder+"/cate_cr_"+str(slot_id)+".txt",'w') as f:
        i_pre = slot_data_np[0,0]
        k = 0
        for x in slot_data_np:
            i_ = x[0]
            u_ = x[1]
            if i_ != i_pre or k == 0:
                i_pre = i_
                if k>0:
                    f.write('\n')
                f.write(str(i_))
                k = 1
            f.write(" " + str(u_))
file_name = args.oriDataset + 'Dis.pkl'

if not os.path.isfile("./"+path_folder+"/" + args.oriDataset + 'Dis.pkl'):
    dis = calDis(data_train)
    with open(os.path.join("./"+path_folder, file_name), 'wb') as file:
        pickle.dump(dis, file)
    file.close()
else:
    with open(os.path.join("./"+path_folder, file_name), 'rb') as file:
        dis = pickle.load(file)
    file.close()

for slot_id in train_slots:
    slot_data = data_train[data_train['time_slot'].isin([slot_id])]

    for location1 in dis.keys():
        nearby_locations = [location2 for location2, distance in dis[location1].items() if distance < 10]
        # 上面的 your_threshold_distance 是你希望的地点距离阈值

        # 筛选出在时间段 slot_id 内的地点
        slot_data_filtered = slot_data[slot_data['pid'].isin(nearby_locations)]
        users = slot_data_filtered['uid']

        if len(users) > 0:
            with open("./"+path_folder+"/s_cr_"+str(slot_id)+".txt",'a') as f:
                f.write(str(location1))
                for user in users:
                    f.write(' ' + str(user))
                f.write('\n')
item_num = item.shape[0]
n_item = item_num
pop_item = []
for i in train_slots:
    path = "./"+path_folder+"/cr_{}.txt".format(i)
    total = 0
    item_pop_list_t=[]
    with open(path) as f:
        for line in f:
            line = line.strip().split()
            item, pop = int(line[0]), len(line[1:])
            item_pop_list_t.append((item,pop))
            total+=pop
    pop_item.append([1/(total+n_item) for _ in range(n_item)])
    # pop_item.append([0/(total) for _ in range(n_item)])
    for item,pop in item_pop_list_t:
        # print(item,n_item)
        pop_item[i][item-1] = (pop+1.0)/(total+n_item)
        # pop_item[i][item] = 1e6*(pop)/(total)
pop_item = np.array(pop_item)
for k in range(pop_item.shape[0]):
    pop_item[k] = (pop_item[k] - pop_item[k].min()) / (pop_item[k].max() - pop_item[k].min())

with open("./"+path_folder+"/item_pop_seq_ori.txt","w") as f:
    for i in range(n_item):
        pop_seq_i = pop_item[:, i]
        write_str = ""
        write_str += str(i+1) + ' '
        for pop in pop_seq_i:
            write_str += str(pop) + ' '
        write_str = write_str.strip(' ')
        write_str += '\n'
        f.write(write_str)

n_cate = cate.shape[0]
pop_item = []
for i in train_slots:
    path = "./"+path_folder+"/cate_cr_{}.txt".format(i)
    total = 0
    item_pop_list_t=[]
    with open(path) as f:
        for line in f:
            line = line.strip().split()
            item, pop = int(line[0]), len(line[1:])
            item_pop_list_t.append((item,pop))
            total+=pop
    pop_item.append([1/(total+n_cate) for _ in range(n_cate)])
    # pop_item.append([0/(total) for _ in range(n_item)])
    for item,pop in item_pop_list_t:
        # print(item,n_item)
        pop_item[i][item-1] = (pop+1.0)/(total+n_cate)
        # pop_item[i][item] = 1e6*(pop)/(total)
pop_item = np.array(pop_item)
for k in range(pop_item.shape[0]):
    pop_item[k] = (pop_item[k] - pop_item[k].min()) / (pop_item[k].max() - pop_item[k].min())

with open("./"+path_folder+"/cate_pop_seq_ori.txt","w") as f:
    for i in range(n_cate):
        pop_seq_i = pop_item[:, i]
        write_str = ""
        write_str += str(i+1) + ' '
        for pop in pop_seq_i:
            write_str += str(pop) + ' '
        write_str = write_str.strip(' ')
        write_str += '\n'
        f.write(write_str)

n_item = item_num
pop_item = []
for i in train_slots:
    path = "./"+path_folder+"/s_cr_{}.txt".format(i)
    total = 0
    item_pop_list_t=[]
    with open(path) as f:
        for line in f:
            line = line.strip().split()
            item, pop = int(line[0]), len(line[1:])
            item_pop_list_t.append((item,pop))
            total+=pop
    pop_item.append([1/(total+n_item) for _ in range(n_item)])
    # pop_item.append([0/(total) for _ in range(n_item)])
    for item,pop in item_pop_list_t:
        # print(item,n_item)
        pop_item[i][item-1] = (pop+1.0)/(total+n_item)
        # pop_item[i][item] = 1e6*(pop)/(total)
pop_item = np.array(pop_item)
for k in range(pop_item.shape[0]):
    pop_item[k] = (pop_item[k] - pop_item[k].min()) / (pop_item[k].max() - pop_item[k].min())

with open("./"+path_folder+"/space_pop_seq_ori.txt","w") as f:
    for i in range(n_item):
        pop_seq_i = pop_item[:, i]
        write_str = ""
        write_str += str(i+1) + ' '
        for pop in pop_seq_i:
            write_str += str(pop) + ' '
        write_str = write_str.strip(' ')
        write_str += '\n'
        f.write(write_str)