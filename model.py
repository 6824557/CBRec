import numpy as np
import torch
import math
import random
from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
from utils import *
import copy
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num,cate_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.cate_num = cate_num
        self.dev = args.device
        self.lmdcate = args.lmdcate
        self.lmdpop = args.lmdpop
        self.methodcate = args.methodcate
        self.methodpop = args.methodpop
        self.maxlen = args.maxlen
        self.lmdattention1 = args.lmdattention1
        self.lmdattention2 = args.lmdattention2
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        #torch.nn.Embedding：随机初始化词向量，词向量值在正态分布N(0,1)中随机取值。
        # padding_idx=None,– 填充id，比如，输入长度为100，但是每次的句子长度并不一样，后面就需要用统一的数字填充，而这里就是指定这个数字，这样，网络在遇到填充id时，就不会计算其与其它符号的相关性。
        self.pos_emb = torch.nn.Embedding(args.maxlen*2, args.hidden_units) # TO IMPROVE
        #添加embedding
        self.cate_emb = torch.nn.Embedding(self.cate_num + 1, args.hidden_units, padding_idx=0)
        self.pop_emb = torch.nn.Embedding(10000+1, args.hidden_units, padding_idx=0)
        self.time_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)

        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)#Dropout方法是一种在训练模型时被广泛应用的trick，目的是防止模型过拟合，原理是使网络中某一层的每个参数以一定概率被mask（变为0），只用剩下的参数进行训练，从而达到防止模型过拟合的目的

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)#而LayerNorm是对整个输入做了归一化，是在样本粒度层面的

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        ###########新attention
        self.attention_layernorms2 = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers2 = torch.nn.ModuleList()
        self.forward_layernorms2 = torch.nn.ModuleList()
        self.forward_layers2 = torch.nn.ModuleList()

        self.last_layernorm2 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)  # 而LayerNorm是对整个输入做了归一化，是在样本粒度层面的

        for _ in range(args.num_blocks):
            new_attn_layernorm2 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms2.append(new_attn_layernorm2)

            new_attn_layer2 = torch.nn.MultiheadAttention(args.hidden_units,
                                                         args.num_heads,
                                                         args.dropout_rate)
            self.attention_layers2.append(new_attn_layer2)

            new_fwd_layernorm2 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms2.append(new_fwd_layernorm2)

            new_fwd_layer2 = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers2.append(new_fwd_layer2)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()
        print("x")
    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats
    def log2featslb(self, log_seqs,cate_seqs,pop_seqs):

        # 计算seq特征
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs2 = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs2 *= self.item_emb.embedding_dim ** 0.5
        a = log_seqs.shape

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))

        cate = self.cate_emb(torch.LongTensor(cate_seqs).to(self.dev))
        cate *= self.cate_emb.embedding_dim ** 0.5
        cate *= self.lmdcate

        pop = self.pop_emb(torch.LongTensor(pop_seqs).to(self.dev))
        pop *= self.pop_emb.embedding_dim ** 0.5
        pop *= self.lmdpop

        if self.methodcate == 1:
            seqs2 += cate
        elif self.methodcate == 2:
            seqs2 *= cate
        elif self.methodcate == 3:
            seqs2 = torch.cat((seqs, cate), dim=0)

        if self.methodcate == 1:
            seqs2 += pop
        elif self.methodcate == 2:
            seqs2 *= pop
        elif self.methodcate == 3:
            seqs2 = torch.cat((seqs, pop), dim=0)
        seqs2 += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)
        seqs2 = self.emb_dropout(seqs2)
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim
        seqs2 *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim
        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats1 = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        for i in range(len(self.attention_layers2)):
            seqs2 = torch.transpose(seqs2, 0, 1)
            Q = self.attention_layernorms2[i](seqs2)
            mha_outputs, _ = self.attention_layers2[i](Q, seqs2, seqs2,
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs2 = Q + mha_outputs
            seqs2 = torch.transpose(seqs2, 0, 1)

            seqs2 = self.forward_layernorms2[i](seqs2)
            seqs2 = self.forward_layers2[i](seqs2)
            seqs2 *=  ~timeline_mask.unsqueeze(-1)
        log_feats2 = self.last_layernorm2(seqs2)

        log_feats = self.lmdattention1 * log_feats1 + self.lmdattention2 * log_feats2
        return log_feats

    # def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, cate_seqs, time_seqs): # for training
    #
    #
    #     log_feats = self.log2featswithTimeCate(log_seqs, cate_seqs) # user_ids hasn't been used yet
    #
    #     pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
    #     neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
    #
    #     pos_logits = (log_feats * pos_embs).sum(dim=-1)
    #     neg_logits = (log_feats * neg_embs).sum(dim=-1)
    #
    #     # pos_pred = self.pos_sigmoid(pos_logits)
    #     # neg_pred = self.neg_sigmoid(neg_logits)
    #
    #     return pos_logits, neg_logits, log_feats, pos_embs, neg_embs

    #lb
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, cate_seqs, pop_seqs):  # for training

        log_feats = self.log2featslb(log_seqs,cate_seqs, pop_seqs)  # user_ids hasn't been used yet
        # log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet
        #计算相似度
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits, log_feats, pos_embs, neg_embs

    def predict(self, user_ids, log_seqs, cate_seqs, pop_seqs, item_indices,  pop_seqs_for_ranklist): # for inference
        # 计算seq特征
        log_feats = self.log2featslb(log_seqs,cate_seqs, pop_seqs)  # user_ids hasn't been used yet


        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)
        item_pop = torch.tensor(pop_seqs_for_ranklist, dtype=torch.float, device=self.dev)
        item_embs = item_embs * item_pop.view(-1, 1)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)


class CounterfactualCL(torch.nn.Module):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, user_num, item_num, cate_num, args):
        super(CounterfactualCL, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.augmethod = args.augmethod
        self.dev = args.device
        self.dev = args.device
        self.batch_size = args.batch_size
        self.tot_judge_hot = args.tot_judge_hot
        self.item_judge_hot = args.tot_judge_hot
        self.cate_judge_hot = args.cate_judge_hot
        self.space_judge_hot = args.space_judge_hot
        self.counter_pro = args.counter_pro
        self.maxlen = args.maxlen
        self.loss_type = 'BPR'

        self.trm_encoder = SASRec(user_num, item_num,cate_num, args).to(args.device)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.mask_default = self.mask_correlated_samples(batch_size=args.batch_size)
        self.nce_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)
        for name, param in self.trm_encoder.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)  # 防止梯度消失的参数初始化
            except:
                pass  # just ignore those failed init layers


    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=1)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask



    def augment(self, item_seq, itemSim, userIntent, user, seq_cate, seq_time, tot_pop_judge, item_pop_judge, cate_pop_judge, space_pop_judge,check_matrix):
        aug_seq1 = []
        aug_len1 = []
        aug_seq2 = []
        aug_len2 = []
        aug_seq = []
        item_idx = [t for t in range(1, self.item_num + 1)]
        item_pop = [check_matrix[t - 1] for t in range(1, self.item_num + 1)]
        for idx, seq in enumerate(item_seq):
            if len(seq) > 1:
                switch = random.sample(range(3), k=2)
                userIntentItem = userIntent[user[idx]]
                counterfactual = random.random()
            else:
                switch = [3, 3]
                aug_seq = seq
            if self.augmethod == 0:
                if counterfactual < self.counter_pro-0.3 :
                    aug_seq = self.item_replace(seq, userIntentItem, itemSim, seq_cate[idx], seq_time[idx], item_pop_judge, tot_pop_judge,item_idx,item_pop )
                elif counterfactual < self.counter_pro-0.1 :
                    aug_seq = self.cate_replace(seq, userIntentItem, itemSim, seq_cate[idx], seq_time[idx], cate_pop_judge, tot_pop_judge,item_idx,item_pop)
                elif counterfactual < self.counter_pro :
                    aug_seq = self.space_replace(seq, userIntentItem, itemSim, seq_cate[idx], seq_time[idx], space_pop_judge, tot_pop_judge,item_idx,item_pop)
                elif counterfactual < self.counter_pro :
                    aug_seq = self.item_delete(seq, userIntentItem, itemSim, seq_cate[idx], seq_time[idx], tot_pop_judge)
                else:
                    aug_seq = self.item_add(seq, userIntentItem, itemSim, seq_cate[idx], seq_time[idx],
                                               tot_pop_judge)
            elif self.augmethod == 1:
                if counterfactual < self.counter_pro-0.3 :
                    aug_seq = self.item_replace(seq, userIntentItem, itemSim, seq_cate[idx], seq_time[idx], item_pop_judge, tot_pop_judge,item_idx,item_pop )
                elif counterfactual < self.counter_pro-0.1 :
                    aug_seq = self.cate_replace(seq, userIntentItem, itemSim, seq_cate[idx], seq_time[idx], cate_pop_judge, tot_pop_judge,item_idx,item_pop)
                elif counterfactual < self.counter_pro :
                    aug_seq = self.space_replace(seq, userIntentItem, itemSim, seq_cate[idx], seq_time[idx], space_pop_judge, tot_pop_judge,item_idx,item_pop)
            elif self.augmethod == 2:
                aug_seq = self.item_delete(seq, userIntentItem, itemSim, seq_cate[idx], seq_time[idx], tot_pop_judge)
            elif self.augmethod == 3:
                aug_seq = self.item_add(seq, userIntentItem, itemSim, seq_cate[idx], seq_time[idx], tot_pop_judge)
            # if counterfactual < self.counter_pro-0.3 :
            #     aug_seq = self.item_replace(seq, userIntentItem, itemSim, seq_cate[idx], seq_time[idx], item_pop_judge, tot_pop_judge,item_idx,item_pop )
            # elif counterfactual < self.counter_pro-0.1 :
            #     aug_seq = self.cate_replace(seq, userIntentItem, itemSim, seq_cate[idx], seq_time[idx], cate_pop_judge, tot_pop_judge,item_idx,item_pop)
            # elif counterfactual < self.counter_pro :
            #     aug_seq = self.space_replace(seq, userIntentItem, itemSim, seq_cate[idx], seq_time[idx], space_pop_judge, tot_pop_judge,item_idx,item_pop)
            # elif counterfactual < self.counter_pro :
            #     aug_seq = self.item_delete(seq, userIntentItem, itemSim, seq_cate[idx], seq_time[idx], tot_pop_judge)


            aug_seq1.append(aug_seq)

            if switch[1] == 0:  # Add
                aug_seq = self.item_crop(seq, len(seq) )
            elif switch[1] == 1:  # Replace
                aug_seq = self.item_mask(seq, len(seq) )
            elif switch[1] == 2:  # Delete
                aug_seq = self.item_reorder(seq, len(seq) )

            aug_seq2.append(aug_seq)

        maxlen1 = max(len(sublist) for sublist in aug_seq1)
        maxlen2 = max(len(sublist) for sublist in aug_seq2)
        aug_seq1 = self.align(aug_seq1, max(maxlen1, maxlen2))
        aug_seq2 = self.align(aug_seq2, max(maxlen1, maxlen2))
        return torch.stack(aug_seq1), torch.stack(aug_seq2)

    def align(self, list, max_length):

        new_list = []

        # 遍历每个子列表，将长度补足到最大长度
        for sublist in list:
            while len(sublist) < max_length:
                sublist.insert(0, 0)
            new_list.append(torch.tensor(sublist, dtype=torch.long))

        return new_list

    def get_top_n_similar_items(self, itemSim, item_id, n, order):
        if item_id not in itemSim:
            return []  # 如果给定的商品ID不在字典中，返回空列表

        # 获取与商品i相似度最高的n个商品ID
        similar_items = sorted(itemSim[item_id].items(), key=lambda x: x[1], reverse=order)[:n]

        # 仅返回商品ID，而不包括相似度值
        top_n_similar_item_ids = [item[0] for item in similar_items]

        return top_n_similar_item_ids




    def item_replace(self, item_seq, userIntentItem, itemSim, seq_cate, seq_time,item_pop_judge,tot_pop_judge,item_idx,item_pop):
        pos_example = []
        random_number = random.random()
        for id, item in enumerate(item_seq):
            hotRate = item_pop_judge[item - 1]
            judge_hot = self.item_judge_hot
            # if type == 0:
            #     hotRate = tot_pop_judge[item - 1]
            #     judge_hot = self.item_judge_hot
            # if type == 1:
            #     hotRate = tot_pop_judge[seq_cate[id]-1]
            #     judge_hot = self.cate_judge_hot
            # if type == 2:
            #     hotRate = tot_pop_judge[item - 1]
            #     judge_hot = self.space_judge_hot
            # if type == 3:
            #     hotRate = tot_pop_judge[item - 1]
            #     judge_hot = self.tot_judge_hot
            if hotRate <= judge_hot or item == 0 :
                pos_example.append(item)
            else:
                if np.count_nonzero(item_seq[:id]) <= self.maxlen:
                    ####相似度
                    if random_number <= 0.5:
                        rep_items = self.get_top_n_similar_items(itemSim, item, 20, True)
                        rep_item = random.choice(rep_items)
                        pos_example.append(rep_item)
                    elif random_number <= 0.8:
                        rep_items = self.get_top_n_similar_items(itemSim, item, 20, False)
                        rep_item = random.choice(rep_items)
                        pos_example.append(rep_item)
                    else:
                        pos_example.append(item)
                else:
                    ###预测
                    if random_number <= 0.7:
                        self.trm_encoder.eval()
                        if len(item_seq[:id]) <= self.maxlen:
                            seq = item_seq[:id]
                            cate = seq_cate[:id]
                        else:
                            seq = item_seq[id - self.maxlen:id]
                            cate = seq_cate[id - self.maxlen: id]
                        seq = np.pad(seq, (max(0, self.maxlen - len(seq)), 0), 'constant')
                        cate = np.pad(cate, (max(0, self.maxlen - len(cate)), 0), 'constant')
                        seq_pop = seq2pop(seq, tot_pop_judge)
                        cc = [np.array(l) for l in [[0], [seq], [cate], [seq_pop], item_idx, item_pop]]
                        predictions = self.trm_encoder.predict(*cc)
                        rank_ls = torch.topk(predictions, 20).indices
                        rank_ls = rank_ls.tolist()[0]
                        rep_item = rank_ls[0]+1
                        pos_example.append(rep_item)
                        self.trm_encoder.train()
                    else:
                        pos_example.append(item)
        return pos_example
        # return torch.tensor(pos_example, dtype=torch.long, device=self.dev)

    def cate_replace(self, item_seq, userIntentItem, itemSim, seq_cate, seq_time,cate_pop_judge, tot_pop_judge,item_idx,item_pop):
        pos_example = []
        random_number = random.random()
        for id, item in enumerate(item_seq):
            hotRate = cate_pop_judge[seq_cate[id]-1]
            judge_hot = self.cate_judge_hot
            # if type == 0:
            #     hotRate = tot_pop_judge[item - 1]
            #     judge_hot = self.item_judge_hot
            # if type == 1:
            #     hotRate = tot_pop_judge[seq_cate[id]-1]
            #     judge_hot = self.cate_judge_hot
            # if type == 2:
            #     hotRate = tot_pop_judge[item - 1]
            #     judge_hot = self.space_judge_hot
            # if type == 3:
            #     hotRate = tot_pop_judge[item - 1]
            #     judge_hot = self.tot_judge_hot
            if hotRate <= judge_hot or item == 0:
                pos_example.append(item)
            else:
                if np.count_nonzero(item_seq[:id]) <= self.maxlen:
                    ####相似度
                    if random_number <= 0.5:

                        rep_items = self.get_top_n_similar_items(itemSim, item, 20, True)
                        rep_item = random.choice(rep_items)
                        pos_example.append(rep_item)
                    elif random_number <= 0.8:
                        rep_items = self.get_top_n_similar_items(itemSim, item, 20, False)
                        rep_item = random.choice(rep_items)
                        pos_example.append(rep_item)
                    else:
                        pos_example.append(item)
                else:
                    # ###预测
                    if random_number <= 0.7:
                        self.trm_encoder.eval()
                        if len(item_seq[:id]) <= self.maxlen:
                            seq = item_seq[:id]
                            cate = seq_cate[:id]
                        else:
                            seq = item_seq[id - self.maxlen: id]
                            cate = seq_cate[id - self.maxlen: id]
                        seq = np.pad(seq, (max(0, self.maxlen - len(seq)), 0), 'constant')
                        cate = np.pad(cate, (max(0, self.maxlen - len(cate)), 0), 'constant')
                        seq_pop = seq2pop(seq, tot_pop_judge)
                        cc = [np.array(l) for l in [[0], [seq], [cate], [seq_pop], item_idx, item_pop]]
                        predictions = self.trm_encoder.predict(*cc)
                        rank_ls = torch.topk(predictions, 20).indices
                        rank_ls = rank_ls.tolist()[0]
                        rep_item = rank_ls[0]+1
                        pos_example.append(rep_item)
                        self.trm_encoder.train()
                    else:
                        pos_example.append(item)
        return pos_example

    def space_replace(self, item_seq, userIntentItem, itemSim, seq_cate, seq_time,space_pop_judge, tot_pop_judge,item_idx,item_pop):
        pos_example = []
        random_number = random.random()
        for id, item in enumerate(item_seq):
            hotRate = space_pop_judge[item - 1]
            judge_hot = self.space_judge_hot
            # if type == 0:
            #     hotRate = tot_pop_judge[item - 1]
            #     judge_hot = self.item_judge_hot
            # if type == 1:
            #     hotRate = tot_pop_judge[seq_cate[id]-1]
            #     judge_hot = self.cate_judge_hot
            # if type == 2:
            #     hotRate = tot_pop_judge[item - 1]
            #     judge_hot = self.space_judge_hot
            # if type == 3:
            #     hotRate = tot_pop_judge[item - 1]
            #     judge_hot = self.tot_judge_hot
            if hotRate <= judge_hot or item == 0:
                pos_example.append(item)
            else:
                if np.count_nonzero(item_seq[:id]) <= self.maxlen:
                    ###相似度
                    if random_number <= 0.5:

                        rep_items = self.get_top_n_similar_items(itemSim, item, 20, True)
                        rep_item = random.choice(rep_items)
                        pos_example.append(rep_item)
                    elif random_number <= 0.8:
                        rep_items = self.get_top_n_similar_items(itemSim, item, 20, False)
                        rep_item = random.choice(rep_items)
                        pos_example.append(rep_item)
                    else:
                        pos_example.append(item)
                else:
                    ###预测
                    if random_number <= 0.7:
                        self.trm_encoder.eval()
                        if len(item_seq[:(id + 1)]) <= self.maxlen:
                            seq = item_seq[:(id + 1)]
                            cate = seq_cate[:(id + 1)]
                        else:
                            seq = item_seq[id - self.maxlen: id]
                            cate = seq_cate[id - self.maxlen: id]
                        seq = np.pad(seq, (max(0, self.maxlen - len(seq)), 0), 'constant')
                        cate = np.pad(cate, (max(0, self.maxlen - len(cate)), 0), 'constant')
                        seq_pop = seq2pop(seq, tot_pop_judge)
                        cc = [np.array(l) for l in [[0], [seq], [cate], [seq_pop], item_idx, item_pop]]
                        predictions = self.trm_encoder.predict(*cc)
                        rank_ls = torch.topk(predictions, 20).indices
                        rank_ls = rank_ls.tolist()[0]
                        rep_item = rank_ls[0]+1
                        pos_example.append(rep_item)
                        self.trm_encoder.train()
                    else:
                        pos_example.append(item)
        return pos_example

    def item_delete(self, item_seq, userIntentItem, itemSim, seq_cate, seq_time, tot_pop_judge):
        pos_example = []
        random_number = random.random()
        for id, item in enumerate(item_seq):
            hotRate = tot_pop_judge[item - 1]
            judge_hot = self.tot_judge_hot
            if hotRate <= judge_hot or item == 0 :
                pos_example.append(item)
            else:
                if random_number <= 0.6:
                    continue
                else:
                    pos_example.append(item)
        return pos_example
        # return torch.tensor(pos_example, dtype=torch.long, device=self.dev)
    def item_add(self, item_seq, userIntentItem, itemSim, seq_cate, seq_time, tot_pop_judge):
        pos_example = []
        random_number = random.random()
        seqlen = len(item_seq)
        for id,item in enumerate(item_seq):
            if id+1 >= seqlen:
                break
            item1 = item_seq[id]
            item2 = item_seq[id+1]
            hotRate1 = tot_pop_judge[item1 - 1]
            hotRate2 = tot_pop_judge[item2 - 1]
            if hotRate1 >= self.tot_judge_hot and hotRate2 >= self.tot_judge_hot and item != 0:

                if random_number <= 0.5:
                    pos_example.append(item)
                elif random_number <= 0.8:
                    rep_items = self.get_top_n_similar_items(itemSim, item, 20, True)
                    rep_item = random.choice(rep_items)
                    pos_example.append(item)
                    pos_example.append(rep_item)

                else:
                    rep_items = self.get_top_n_similar_items(itemSim, item, 20, False)
                    rep_item = random.choice(rep_items)
                    pos_example.append(item)
                    pos_example.append(rep_item)

            else:
                pos_example.append(item)
        return pos_example
    def item_crop(self, item_seq, item_seq_len, eta=0.6):
        num_left = math.floor(item_seq_len * eta)
        crop_begin = random.randint(0, item_seq_len - num_left)
        croped_item_seq = [0] * len(item_seq)
        if crop_begin + num_left < len(item_seq):
            croped_item_seq[:num_left] = item_seq[crop_begin:crop_begin + num_left]
        else:
            croped_item_seq[:num_left] = item_seq[crop_begin:]
        return croped_item_seq

    def item_mask(self, item_seq, item_seq_len, gamma=0.3):
        num_mask = math.floor(item_seq_len * gamma)
        mask_index = random.sample(range(item_seq_len), k=num_mask)

        masked_item_seq = item_seq.copy()  # 使用 Python 列表的 copy 方法

        # 假设 self.n_items 是一个用于语义掩码的特殊标记
        for index in mask_index:
            masked_item_seq[index] = 0
        masked_item_seq = masked_item_seq.tolist()
        return masked_item_seq

    def item_reorder(self, item_seq, item_seq_len, beta=0.6):
        num_reorder = math.floor(item_seq_len * beta)
        reorder_begin = random.randint(0, item_seq_len - num_reorder)

        reordered_item_seq = item_seq.copy()  # 使用 Python 列表的 copy 方法

        reorder_index = list(range(reorder_begin, reorder_begin + num_reorder))
        random.shuffle(reorder_index)

        # 创建一个临时列表以保存重新排序的元素
        temp_reordered = [item_seq[i] for i in reorder_index]

        # 使用临时列表的元素替换原始列表中的相应位置
        for i in range(num_reorder):
            reordered_item_seq[reorder_begin + i] = temp_reordered[i]
        reordered_item_seq = reordered_item_seq.tolist()
        return reordered_item_seq

    def forward(self, item_seq):
        # position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        # position_embedding = self.position_embedding(position_ids)
        #
        # item_emb = self.item_embedding(item_seq)
        # input_emb = item_emb + position_embedding
        # input_emb = self.LayerNorm(input_emb)
        # input_emb = self.dropout(input_emb)
        #
        # extended_attention_mask = self.get_attention_mask(item_seq)
        #
        # trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        # output = trm_output[-1]
        # output = self.gather_indexes(output, item_seq_len - 1)
        log_feats = self.trm_encoder.log2feats(item_seq)
        return log_feats  # [B H]

    # def calculate_loss(self, item_seq, itemSim, userIntent, user):
        # item_seq = interaction[self.ITEM_SEQ]
        # item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # seq_output = self.forward(item_seq, item_seq_len)
        # pos_items = interaction[self.POS_ITEM_ID]
        # if self.loss_type == 'BPR':
        #     neg_items = interaction[self.NEG_ITEM_ID]
        #     pos_items_emb = self.item_embedding(pos_items)
        #     neg_items_emb = self.item_embedding(neg_items)
        #     pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
        #     neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
        #     loss = self.loss_fct(pos_score, neg_score)
        # else:  # self.loss_type = 'CE'
        #     test_item_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
        #     logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        #     loss = self.loss_fct(logits, pos_items)

        # NCE
        # aug_item_seq1, aug_item_seq2= self.augment(item_seq, itemSim, userIntent, user)
        # aug_item_seq1, aug_len1, aug_item_seq2, aug_len2 = \
        #     interaction['aug1'], interaction['aug_len1'], interaction['aug2'], interaction['aug_len2']
        # seq_output1 = self.forward(aug_item_seq1, aug_len1)
        # seq_output2 = self.forward(aug_item_seq2, aug_len2)
        #
        # nce_logits, nce_labels = self.info_nce(seq_output1, seq_output2, temp=self.tau, batch_size=aug_len1.shape[0],
        #                                        sim=self.sim)

        # nce_logits = torch.mm(seq_output1, seq_output2.T)
        # nce_labels = torch.tensor(list(range(nce_logits.shape[0])), dtype=torch.long, device=item_seq.device)

        # with torch.no_grad():
        #     alignment, uniformity = self.decompose(seq_output1, seq_output2, seq_output,
        #                                            batch_size=item_seq_len.shape[0])
        #
        # nce_loss = self.nce_fct(nce_logits, nce_labels)
        #
        # return loss + self.lmd * nce_loss, alignment, uniformity

    def decompose(self, z_i, z_j, origin_z, batch_size):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        # pairwise l2 distace
        sim = torch.cdist(z, z, p=2)

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        alignment = positive_samples.mean()

        # pairwise l2 distace
        sim = torch.cdist(origin_z, origin_z, p=2)
        mask = torch.ones((batch_size, batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        negative_samples = sim[mask].reshape(batch_size, -1)
        uniformity = torch.log(torch.exp(-2 * negative_samples).mean())

        return alignment, uniformity

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diagonal(sim, offset=batch_size)
        sim_j_i = torch.diagonal(sim, offset=-batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores