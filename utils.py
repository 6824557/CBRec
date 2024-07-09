import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import math
import gensim
from tqdm import tqdm
from collections import Counter
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t
#data load
def data_partition(dir):
    usernum = 0
    itemnum = 0
    catenum = 0
    User = defaultdict(list)
    UserCate = defaultdict(list)
    UserTime = defaultdict(list)
    user_train_cate = {}
    user_train_time = {}
    user_train = {}
    user_valid = {}
    user_test = {}
    itemCate = {}
    itemPos = {}
    point2cate = {}
    # assume user/item index starting from 1
    f = open(dir+'/train_with_time.txt' , 'r')
    for line in f:
        a = line.rstrip().split(' ')
        u, i, c, _, lan, lon, timestap, _,_,_,_ = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        c = int(float(c))
        lan = float(lan)
        lon = float(lon)
        timestap = float(timestap)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        catenum = max(c, catenum)
        User[u].append(i)
        UserCate[u].append(c)
        UserTime[u].append(timestap)
        itemCate[i] = c
        itemPos[i] = (lan,lon)
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_train_cate[user] = UserCate[user]
            user_train_time[user] = UserTime[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_train_cate[user] = UserCate[user][:-2]
            user_train_time[user] = UserTime[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum,catenum, user_train_cate, user_train_time,itemCate,itemPos]

def data_partitionSparisty(dir, sample_fraction=0.5):
    usernum = 0
    itemnum = 0
    catenum = 0
    User = defaultdict(list)
    UserCate = defaultdict(list)
    UserTime = defaultdict(list)
    user_train_cate = {}
    user_train_time = {}
    user_train = {}
    user_valid = {}
    user_test = {}
    itemCate = {}
    itemPos = {}
    point2cate = {}
    # assume user/item index starting from 1
    f = open(dir+'/train_with_time.txt' , 'r')
    for line in f:
        a = line.rstrip().split(' ')
        u, i, c, _, lan, lon, _, timestap = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        c = int(c)
        lan = float(lan)
        lon = float(lon)
        timestap = float(timestap)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        catenum = max(c, catenum)
        User[u].append(i)
        UserCate[u].append(c)
        UserTime[u].append(timestap)
        itemCate[i] = c
        itemPos[i] = (lan,lon)
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_train_cate[user] = UserCate[user]
            user_train_time[user] = UserTime[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            # 对每个用户的签到数据进行随机采样，保留sample_fraction比例的数据
            indices = list(range(nfeedback - 2))
            sampled_indices = sorted(random.sample(indices, int(len(indices) * sample_fraction)))
            user_train[user] = [User[user][i] for i in sampled_indices]
            user_train_cate[user] = [UserCate[user][i] for i in sampled_indices]
            user_train_time[user] = [UserTime[user][i] for i in sampled_indices]

            user_train[user].append(User[user][-2])  # 确保验证集的前一个数据包含在训练集中
            user_train_cate[user].append(UserCate[user][-2])
            user_train_time[user].append(UserTime[user][-2])

            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum, catenum, user_train_cate, user_train_time, itemCate, itemPos]

#calculate similarity
def generate_item_similarity(seq, model_type, usernum, itemnum):
    """
    calculate co-rated users between items
    """
    print("getting item similarity...")

    C = dict()
    N = dict()
    itemSimBest = dict()
    for idx, (u, items) in enumerate(seq.items()):
        if idx%2000 == 12:
            print("proceeded: ", idx)
            break
        if model_type in ['ItemCF', 'ItemCF_IUF']:
            if model_type == 'ItemCF':
                for i in items:
                    N.setdefault(i,0)
                    N[i] += 1
                    for j in items:
                        if i == j:
                            continue
                        C.setdefault(i,{})
                        C[i].setdefault(j,0)
                        C[i][j] += 1
            elif model_type == 'ItemCF_IUF':
                for i in items:
                    N.setdefault(i,0)
                    N[i] += 1
                    for j in items:
                        if i == j:
                            continue
                        C.setdefault(i,{})
                        C[i].setdefault(j,0)
                        C[i][j] += 1 / math.log(1 + len(items) * 1.0)
            for idx, (cur_item, related_items) in enumerate(C.items()):
                if idx%2000 == 0:
                    print("proceeded: ", idx)
                itemSimBest.setdefault(cur_item,{})
                for related_item, score in related_items.items():
                    itemSimBest[cur_item].setdefault(related_item,0)
                    itemSimBest[cur_item][related_item] = score / math.sqrt(N[cur_item] * N[related_item])
        elif model_type == 'Item2Vec':
            list_of_lists = [value for value in seq.values()]
            item2vec_model = gensim.models.Word2Vec(sentences=list_of_lists,
                                                    vector_size=20, window=5, min_count=0,
                                                    epochs=100)
            itemSimBest = dict()
            total_item_nums = len(item2vec_model.wv.index_to_key)
            print("Step 2: convert to item similarity dict")
            total_items = tqdm(item2vec_model.wv.index_to_key, total=total_item_nums)
            for cur_item in total_items:
                related_items = item2vec_model.wv.most_similar(positive=[cur_item], topn=itemnum)
                itemSimBest.setdefault(cur_item, {})
                for (related_item, score) in related_items:
                    itemSimBest[cur_item].setdefault(related_item, 0)
                    itemSimBest[cur_item][related_item] = score
    return itemSimBest

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED, user_train_cate, user_train_time,check_matrix):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        pos_cate = np.zeros([maxlen], dtype=np.int32)
        pos_time = np.zeros([maxlen], dtype=np.int32)
        seq_cate = np.zeros([maxlen], dtype=np.int32)
        seq_time = np.zeros([maxlen], dtype=np.int32)
        pos_pop = np.zeros([maxlen], dtype=float)
        neg_pop = np.zeros([maxlen], dtype=float)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        idx = maxlen - 1
        for i in reversed(user_train_cate[user][:]):
            pos_cate[idx] = i
            idx -= 1
            if idx == -1: break

        idx = maxlen - 1
        for i in reversed(user_train_cate[user][:-1]):
            seq_cate[idx] = i
            idx -= 1
            if idx == -1: break

        idx = maxlen - 1
        for i in reversed(user_train_time[user][:]):
            pos_time[idx] = i
            idx -= 1
            if idx == -1: break

        idx = maxlen - 1
        for i in reversed(user_train_time[user][:-1]):
            seq_time[idx] = i
            idx -= 1
            if idx == -1: break

        for index,value in enumerate(pos):
            if value == 0:
                continue
            pos_pop[index] = check_matrix[pos[index]-1]
            neg_pop[index] = check_matrix[neg[index]-1]


        return (user, seq, pos, neg, seq_cate, seq_time, pos_pop, neg_pop)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, user_train_cate, user_train_time, check_matrix, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      68,
                                                      user_train_cate,
                                                      user_train_time,
                                                      check_matrix
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def load_popularity(path):
    #读取地点签到率
    pop_save_path = path + "/item_pop_seq_ori.txt"
    print("popularity used:",pop_save_path)
    with open(pop_save_path) as f:
        pop_item_all = []
        next(f)
        for line in f:
            item, pop = line.strip().split('\t')
            item = int(item)
            pop = float(pop)
            pop_item_all.append(pop)
    pop_item_all = np.array(pop_item_all)
    print("load pop information:",pop_item_all.mean(),pop_item_all.max(),pop_item_all.min())
    # 读取类型签到率
    pop_save_path = path + "/cate_pop_seq_ori.txt"
    print("popularity used:", pop_save_path)
    with open(pop_save_path) as f:
        pop_cate_all = []
        next(f)
        for line in f:
            item, pop = line.strip().split('\t')
            item = int(item)
            pop = float(pop)
            pop_cate_all.append(pop)
    pop_cate_all = np.array(pop_cate_all)
    # 读取地区签到率
    pop_save_path = path + "/space_pop_seq_ori.txt"
    print("popularity used:", pop_save_path)
    with open(pop_save_path) as f:
        print("pop save path: ", pop_save_path)
        pop_space_all = []
        next(f)
        for line in f:
            item, pop = line.strip().split('\t')
            item = int(item)
            pop = float(pop)
            pop_space_all.append(pop)
    pop_space_all = np.array(pop_space_all)
    print("load pop information:", pop_space_all.mean(), pop_space_all.max(), pop_space_all.min())
    # 读取时间签到率
    pop_save_path = path + "/time_pop_seq_ori.txt"
    print("popularity used:", pop_save_path)
    with open(pop_save_path) as f:
        print("pop save path: ", pop_save_path)
        pop_time_all = []
        next(f)
        flag = 0
        for line in f:
            item, pop = line.strip().split('\t')
            item = int(item)
            if flag == item-1:
                flag = item
            else:
                for i in range(flag,item-1):
                    pop_time_all.append(0)
                    flag = item
            pop = float(pop)
            pop_time_all.append(pop)
    pop_time_all = np.array(pop_time_all)
    print("load pop information:", pop_time_all.mean(), pop_time_all.max(), pop_time_all.min())
    return pop_item_all, pop_cate_all, pop_space_all,pop_time_all

def ave(x):  # 防止数值溢出
    return x / x.sum()

def evaluate(model, dataset, maxlen, check_matrix, tot_pop, item_groups, epoch):
    [train, valid, test, usernum, itemnum,catenum, user_train_cate, user_train_time,itemCate,itemPos] = copy.deepcopy(dataset)
    topk = [1, 5, 10, 20]  # 1,5,10,20
    NDCG = [0.0] * len(topk)
    HR = [0.0] * len(topk)
    ACC = [0.0] * len(topk)
    NDCGT = [0.0] * len(topk)
    RECALL = [0.0] * len(topk)
    valid_user = 0.0
    valid_item = 0.0
    group = 10
    group_num1 = [0]*group
    group_num5 = [0]*group
    group_num10 = [0]*group
    group_num20 = [0]*group
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
        # while users not in train:
        #     users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
        # while users not in train:
        #     users = range(1, usernum + 1)
    pred_label = {}
    wr_test = []
    group_recommendation_count = {group: {'top1': 0, 'top5': 0, 'top10': 0, 'top20': 0} for group in set(item_groups)}
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([maxlen], dtype=np.int32)
        cate = np.zeros([maxlen], dtype=np.int32)
        item_pop1 = []
        item_pop2 =[]
        idx = maxlen - 1
        seq[idx] = valid[u][0]
        cate[idx] = itemCate[seq[idx]]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            cate[idx] = itemCate[seq[idx]]
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        # 原始测试

        item_idx = [test[u][0]]
        item_pop1.append(check_matrix[test[u][0]-1])
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            item_pop1.append(check_matrix[t - 1])
        seq_pop = seq2pop(seq,tot_pop)
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [cate], [seq_pop], item_idx,item_pop1]])
        predictions = predictions[0]  # - for 1st argsort DESC

        #####去偏效果

        top_indices = predictions.argsort()[:20]
        lable1 = 0
        lable2 = 0
        lable3 = 0
        lable4 = 0

        # 统计每个组在前1、5、10、20的推荐中出现的次数
        for i, item_index in enumerate(top_indices):
            group = item_groups[item_index]
            if i < 1 and lable1 == 0:
                group_recommendation_count[group]['top1'] += 1
                lable1 = 1
            if i < 5 and lable2 == 0:
                group_recommendation_count[group]['top5'] += 1
                lable2 = 1
            if i < 10 and lable3 == 0:
                group_recommendation_count[group]['top10'] += 1
                lable3 = 1
            if i < 20 and lable4 == 0:
                group_recommendation_count[group]['top20'] += 1
                lable4 = 1
        ##########去偏效果

        for i, k in enumerate(topk):
            rank = predictions.argsort().argsort()[0].item()
            if rank < k:
                NDCG[i] += 1 / np.log2(rank+1 + 2)
                HR[i] += 1

        item_idx2 = [t for t in range(1, itemnum + 1)]
        item_pop2 = [check_matrix[t - 1] for t in range(1, itemnum + 1)]
        cc = [np.array(l) for l in [[u], [seq], [cate], [seq_pop], item_idx2, item_pop2]]
        predictions2 = model.predict(*cc)
        rank_ls = torch.topk(predictions2, 20).indices
        label = test[u][0]
        valid_user += 1
        valid_item += len([item for item in train[u] if item != 0])
        rank_ls = rank_ls.tolist()[0]
        for i, k in enumerate(topk):
            if label - 1 in rank_ls[:k]:
                ACC[i] += 1
                NDCGT[i] += 1 / np.log2(rank_ls.index(label - 1) + 1 + 2)
            RECALL[i] += sum(1 for item in train[u] if item in rank_ls[:k])
            RECALL[i] += sum(1 for item in valid[u] if item in rank_ls[:k])
            RECALL[i] += sum(1 for item in test[u] if item in rank_ls[:k])
        # if label - 1 in rank_ls[:5]:
        #     ACC[0] += 1
        #     a = rank_ls.index(label - 1) + 1
        #     NDCGT[0] += 1 / np.log2(rank_ls.index(label - 1) + 1 + 2)
        # if label - 1 in rank_ls[:10]:
        #     ACC[1] += 1
        #     NDCGT[1] += 1 / np.log2(rank_ls.index(label - 1) + 1 + 2)
        # if label - 1 in rank_ls[:20]:
        #     ACC[2] += 1
        #     NDCGT[2] += 1 / np.log2(rank_ls.index(label - 1) + 1 + 2)

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    NDCG = [ndcg / valid_user for ndcg in NDCG]
    HR = [hr / valid_user for hr in HR]
    ACC = [acc / valid_user for acc in ACC]
    NDCGT = [ndcgt / valid_user for ndcgt in NDCGT]
    RECALL = [recall / valid_item for recall in RECALL]
    return NDCG, HR, ACC, NDCGT, RECALL,group_recommendation_count

def evaluatelb(model, dataset, maxlen, check_matrix, tot_pop):
    [train, valid, test, usernum, itemnum,catenum, user_train_cate, user_train_time,itemCate,itemPos] = copy.deepcopy(dataset)
    topk = [1, 5, 10, 20]  # 1,5,10,20
    ndcgs = [0.0] * len(topk)
    hits = [0.0] * len(topk)
    recalls = [0.0] * len(topk)
    valid_user = 0.0
    valid_item = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
        # while users not in train:
        #     users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
        # while users not in train:
        #     users = range(1, usernum + 1)
    pred_label = {}
    wr_test = []
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([maxlen], dtype=np.int32)
        cate = np.zeros([maxlen], dtype=np.int32)
        item_pop1 = []
        item_pop2 =[]
        idx = maxlen - 1
        seq[idx] = valid[u][0]
        cate[idx] = itemCate[seq[idx]]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            cate[idx] = itemCate[seq[idx]]
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        rated.add(test[u][0])
        rated.add(valid[u][0])
        # 原始测试
        seq_pop = seq2pop(seq, tot_pop)
        item_idx2 = [t for t in range(1, itemnum + 1)]
        item_pop2 = [check_matrix[t - 1] for t in range(1, itemnum + 1)]
        cc = [np.array(l) for l in [[u], [seq], [cate], [seq_pop], item_idx2, item_pop2]]
        predictions2 = model.predict(*cc)
        rank_ls = torch.topk(predictions2, 20).indices
        valid_user += 1
        valid_item += len([item for item in train[u] if item != 0])
        rank_ls = rank_ls.tolist()[0]
        for k in topk:
            if len(rank_ls) >= k:
                # 获取前 k 个推荐项目
                topk_recommendations = rank_ls[:k]

                # 计算 Recall
                recall = len(set(topk_recommendations) & rated) / len(rated)
                recalls[k].append(recall)

                # 计算 Hit
                hit = 1 if len(set(topk_recommendations) & rated) > 0 else 0
                hits[k].append(hit)

                # 计算 NDCG
                dcg = 0.0
                for i in range(k):
                    item = topk_recommendations[i]
                    if item in rated:
                        dcg += 1 / np.log2(i + 2)  # i+2因为索引是从0开始
                idcg = 0.0
                num_relevant_items = min(len(rated), k)  # 获取实际交互项目的数量和 k 中较小的一个
                for i in range(num_relevant_items):
                    idcg += 1 / np.log2(i + 2)
                if idcg == 0:
                    ndcg = 0.0
                else:
                    ndcg = dcg / idcg
                ndcgs[k].append(ndcg)

        # 计算每个指标的平均值


        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    average_recalls = [ np.mean(v) for k, v in recalls.items()]
    average_hits = [ np.mean(v) for k, v in hits.items()]
    average_ndcgs = [ np.mean(v) for k, v in ndcgs.items()]
    return average_recalls,average_hits,average_ndcgs

# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum, user_train_cate, user_train_time] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

def seq2pop(seq, ItemPop):
    pop = []
    for item in seq:
        pop.append(ItemPop[item-1]*10000)
    pop = [arr.astype(int) for arr in seq]
    return pop


def gini_coefficient(x):
    """Calculate the Gini coefficient of a numpy array."""
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))
    return gini


def evaluatefin(model, dataset, maxlen, check_matrix, tot_pop, item_groups, epoch):
    [train, valid, test, usernum, itemnum, catenum, user_train_cate, user_train_time, itemCate,
     itemPos] = copy.deepcopy(dataset)
    topk = [1, 5, 10, 20]
    NDCG = [0.0] * len(topk)
    HR = [0.0] * len(topk)
    ACC = [0.0] * len(topk)
    NDCGT = [0.0] * len(topk)
    RECALL = [0.0] * len(topk)
    valid_user = 0.0
    valid_item = 0.0

    # 初始化覆盖率和基尼系数相关变量
    all_recommended_items = []

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    group_recommendation_count = {group: {'top1': 0, 'top5': 0, 'top10': 0, 'top20': 0} for group in set(item_groups)}

    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([maxlen], dtype=np.int32)
        cate = np.zeros([maxlen], dtype=np.int32)
        item_pop1 = []
        item_pop2 = []
        idx = maxlen - 1
        seq[idx] = valid[u][0]
        cate[idx] = itemCate[seq[idx]]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            cate[idx] = itemCate[seq[idx]]
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)

        item_idx = [test[u][0]]
        item_pop1.append(check_matrix[test[u][0] - 1])
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            item_pop1.append(check_matrix[t - 1])
        seq_pop = seq2pop(seq, tot_pop)
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [cate], [seq_pop], item_idx, item_pop1]])
        predictions = predictions[0]

        top_indices = predictions.argsort()[:20]

        # 记录被推荐的物品
        all_recommended_items.extend([item_idx[i] for i in top_indices])

        lable1, lable2, lable3, lable4 = 0, 0, 0, 0
        for i, item_index in enumerate(top_indices):
            group = item_groups[item_index]
            if i < 1 and lable1 == 0:
                group_recommendation_count[group]['top1'] += 1
                lable1 = 1
            if i < 5 and lable2 == 0:
                group_recommendation_count[group]['top5'] += 1
                lable2 = 1
            if i < 10 and lable3 == 0:
                group_recommendation_count[group]['top10'] += 1
                lable3 = 1
            if i < 20 and lable4 == 0:
                group_recommendation_count[group]['top20'] += 1
                lable4 = 1

        for i, k in enumerate(topk):
            rank = predictions.argsort().argsort()[0].item()
            if rank < k:
                NDCG[i] += 1 / np.log2(rank + 1 + 2)
                HR[i] += 1

        item_idx2 = [t for t in range(1, itemnum + 1)]
        item_pop2 = [check_matrix[t - 1] for t in range(1, itemnum + 1)]
        cc = [np.array(l) for l in [[u], [seq], [cate], [seq_pop], item_idx2, item_pop2]]
        predictions2 = model.predict(*cc)
        rank_ls = torch.topk(predictions2, 20).indices
        label = test[u][0]
        valid_user += 1
        valid_item += len([item for item in train[u] if item != 0])
        rank_ls = rank_ls.tolist()[0]
        for i, k in enumerate(topk):
            if label - 1 in rank_ls[:k]:
                ACC[i] += 1
                NDCGT[i] += 1 / np.log2(rank_ls.index(label - 1) + 1 + 2)
            RECALL[i] += sum(1 for item in train[u] if item in rank_ls[:k])
            RECALL[i] += sum(1 for item in valid[u] if item in rank_ls[:k])
            RECALL[i] += sum(1 for item in test[u] if item in rank_ls[:k])

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    NDCG = [ndcg / valid_user for ndcg in NDCG]
    HR = [hr / valid_user for hr in HR]
    ACC = [acc / valid_user for acc in ACC]
    NDCGT = [ndcgt / valid_user for ndcgt in NDCGT]
    RECALL = [recall / valid_item for recall in RECALL]

    # 计算覆盖率
    unique_recommended_items = len(set(all_recommended_items))
    coverage = unique_recommended_items / itemnum

    # 计算基尼系数
    recommended_item_counts = Counter(all_recommended_items)
    recommended_item_frequencies = np.array(list(recommended_item_counts.values()))
    gini = gini_coefficient(recommended_item_frequencies)

    return NDCG, HR, group_recommendation_count, coverage
def evaluatefindiff(model, dataset, maxlen, check_matrix, tot_pop, item_groups, epoch):
    [train, valid, test, usernum, itemnum, catenum, user_train_cate, user_train_time, itemCate,
     itemPos] = copy.deepcopy(dataset)
    itemnum = 6449
    catenum = 285
    topk = [1, 5, 10, 20]
    NDCG = [0.0] * len(topk)
    HR = [0.0] * len(topk)
    ACC = [0.0] * len(topk)
    NDCGT = [0.0] * len(topk)
    RECALL = [0.0] * len(topk)
    valid_user = 0.0
    valid_item = 0.0

    # 初始化覆盖率和基尼系数相关变量
    all_recommended_items = []

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    group_recommendation_count = {group: {'top1': 0, 'top5': 0, 'top10': 0, 'top20': 0} for group in set(item_groups)}

    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([maxlen], dtype=np.int32)
        cate = np.zeros([maxlen], dtype=np.int32)
        item_pop1 = []
        item_pop2 = []
        idx = maxlen - 1
        seq[idx] = valid[u][0]
        cate[idx] = itemCate[seq[idx]]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            cate[idx] = itemCate[seq[idx]]
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)

        item_idx = [test[u][0]]
        item_pop1.append(check_matrix[test[u][0] - 1])
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            item_pop1.append(check_matrix[t - 1])
        seq_pop = seq2pop(seq, tot_pop)
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [cate], [seq_pop], item_idx, item_pop1]])
        predictions = predictions[0]

        top_indices = predictions.argsort()[:20]

        # 记录被推荐的物品
        all_recommended_items.extend([item_idx[i] for i in top_indices])

        lable1, lable2, lable3, lable4 = 0, 0, 0, 0
        for i, item_index in enumerate(top_indices):
            group = item_groups[item_index]
            if i < 1 and lable1 == 0:
                group_recommendation_count[group]['top1'] += 1
                lable1 = 1
            if i < 5 and lable2 == 0:
                group_recommendation_count[group]['top5'] += 1
                lable2 = 1
            if i < 10 and lable3 == 0:
                group_recommendation_count[group]['top10'] += 1
                lable3 = 1
            if i < 20 and lable4 == 0:
                group_recommendation_count[group]['top20'] += 1
                lable4 = 1

        for i, k in enumerate(topk):
            rank = predictions.argsort().argsort()[0].item()
            if rank < k:
                NDCG[i] += 1 / np.log2(rank + 1 + 2)
                HR[i] += 1

        item_idx2 = [t for t in range(1, itemnum + 1)]
        item_pop2 = [check_matrix[t - 1] for t in range(1, itemnum + 1)]
        cc = [np.array(l) for l in [[u], [seq], [cate], [seq_pop], item_idx2, item_pop2]]
        predictions2 = model.predict(*cc)
        rank_ls = torch.topk(predictions2, 20).indices
        label = test[u][0]
        valid_user += 1
        valid_item += len([item for item in train[u] if item != 0])
        rank_ls = rank_ls.tolist()[0]
        for i, k in enumerate(topk):
            if label - 1 in rank_ls[:k]:
                ACC[i] += 1
                NDCGT[i] += 1 / np.log2(rank_ls.index(label - 1) + 1 + 2)
            RECALL[i] += sum(1 for item in train[u] if item in rank_ls[:k])
            RECALL[i] += sum(1 for item in valid[u] if item in rank_ls[:k])
            RECALL[i] += sum(1 for item in test[u] if item in rank_ls[:k])

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    NDCG = [ndcg / valid_user for ndcg in NDCG]
    HR = [hr / valid_user for hr in HR]
    ACC = [acc / valid_user for acc in ACC]
    NDCGT = [ndcgt / valid_user for ndcgt in NDCGT]
    RECALL = [recall / valid_item for recall in RECALL]

    # 计算覆盖率
    unique_recommended_items = len(set(all_recommended_items))
    coverage = unique_recommended_items / itemnum

    # 计算基尼系数
    recommended_item_counts = Counter(all_recommended_items)
    recommended_item_frequencies = np.array(list(recommended_item_counts.values()))
    gini = gini_coefficient(recommended_item_frequencies)

    return NDCG, HR, ACC, NDCGT, RECALL, group_recommendation_count, coverage, gini