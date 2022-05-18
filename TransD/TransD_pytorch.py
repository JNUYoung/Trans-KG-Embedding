"""
TransD
"""


# import packages
import codecs
import numpy as np
import copy
import time
import random
import matplotlib.pyplot as plt
import json
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
# operator模块输出一系列对应Python内部操作符的函数
import operator
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define global variables
# 实体-实体ID
entities2id = {}
# 关系-关系ID
relations2id = {}
# tails per head : 每个头节点所关联的尾节点数目，相当于反映了该节点的出度
relation_tph = {}
# heads per tail : 每个尾节点所关联的头节点数目，相当于反映了该节点的入度
relation_hpt = {}

# 数据加载函数
def dataloader(trainingset, entityset, relationset, validset):
    """
    读取知识图谱数据集并返回TransD模型规定的数据格式
    :param trainingset: 存储参与训练三元组的txt文件，每行的数据格式为：head_entity  relation    tail_entity
    :param entityset: entity2id.txt，每行的数据格式为：entity    id
    :param relationset: relation2id.txt 每行的数据格式为: relation    id
    :param validset: 验证集三元组txt文件 每行的数据格式为：head_entity  relation    tail_entity
    :return: entity->list, relation->list, triple_list, valid_triple_list
    """
    print("------start extracting EntityList and RelationList------")
    time1 = time.time()
    entity, relation = [], []
    with open(entityset, 'r') as f1, open(relationset, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            entities2id[line[0]] = line[1]
            entity.append(int(line[1]))

        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relations2id[line[0]] = line[1]
            relation.append(int(line[1]))
    time2 = time.time()
    print(f"------finished extracting EntityList and RelationList,time costing: {time2-time1} seconds------")

    triple_list, relation_head, relation_tail = [], {}, {}
    print("------starting extracting TriplesList for training------")
    time3 = time.time()
    with codecs.open(trainingset, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue
            h_ = int(entities2id[triple[0]])
            r_ = int(relations2id[triple[1]])
            t_ = int(entities2id[triple[2]])
            triple_list.append([h_, r_, t_])
            if r_ in relation_head:
                if h_ in relation_head[r_]:
                    relation_head[r_][h_] += 1
                else:
                    relation_head[r_][h_] = 1
            else:
                relation_head[r_] = {}
                relation_head[r_][h_] = 1

            if r_ in relation_tail:
                if t_ in relation_tail[r_]:
                    relation_tail[r_][t_] += 1
                else:
                    relation_tail[r_][t_] = 1
            else:
                relation_tail[r_] = {}
                relation_tail[r_][t_] = 1

    for r_ in relation_head:
        sum1, sum2 = 0, 0
        for head in relation_head[r_]:
            sum1 += 1
            sum2 += relation_head[r_][head]
        tph = sum2 / sum1
        relation_tph[r_] = tph

    for r_ in relation_tail:
        sum1, sum2 = 0, 0
        for tail in relation_tail[r_]:
            sum1 += 1
            sum2 += relation_tail[r_][tail]
        hpt = sum2 / sum1
        relation_hpt[r_] = hpt
    time4 = time.time()
    print(f"------finished extracting TriplesList for training , time costing : {time4-time3} seconds------")

    print("------start extracting TriplesList for validation------")
    time5 = time.time()
    valid_triple_list = []
    with codecs.open(validset, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue
            h_ = int(entities2id[triple[0]])
            r_ = int(relations2id[triple[1]])
            t_ = int(entities2id[triple[2]])
            valid_triple_list.append([h_, r_, t_])
    time6 = time.time()
    print(f"------finished extracting TriplesList for validation , time costing : {time6 - time5} seconds------")
    print(f"Complete All Data Initialization.Details are as below:\nentity_nums:{len(entity)}\nrelation_nums:{len(relation)}\ntraining_triples_nums:{len(triple_list)}\nvalidation_triples_nums:{len(valid_triple_list)}\ntotally_time_cost:{time6-time1}")
    return entity, relation, triple_list, valid_triple_list

# L1 范数
def norm_l1(h, r, t):
    return np.sum(np.fabs(h + r - t))

# L2 范数
def norm_l2(h, r, t):
    return np.sum(np.square(h + r - t))

# TransD模型基类型定义
class TransD_Model(nn.Module):
    def __init__(self, entity_num, relation_num, dim, margin, norm, C):
        """
        :param entity_num:实体数目
        :param relation_num: 关系数目
        :param dim: embedding的维度
        :param margin: 正确三元组和错误的三元组之间的间隔修正，越大对于嵌入词向量的修改就越严格
        :param norm:距离的计算
        :param C:正则化系数
        """
        super(TransD_Model, self).__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.dim = dim
        self.margin = margin
        self.norm = norm
        self.C = C

        # 实体的embedding层
        self.ent_embedding = torch.nn.Embedding(num_embeddings=self.entity_num,embedding_dim=self.dim).to(device)
        # 关系的embedding层
        self.rel_embedding = torch.nn.Embedding(num_embeddings=self.relation_num,embedding_dim=self.dim).to(device)
        # 实体的映射矩阵
        self.ent_transfer = torch.nn.Embedding(num_embeddings=self.entity_num,embedding_dim=self.dim).to(device)
        # 关系的映射矩阵
        self.rel_transfer = torch.nn.Embedding(num_embeddings=self.relation_num,embedding_dim=self.dim).to(device)
        # 损失函数定义
        self.loss_F = nn.MarginRankingLoss(self.margin, reduction="mean").to(device)
        self.__data_init()

    # 参数初始化
    def __data_init(self):
        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        nn.init.xavier_uniform_(self.ent_transfer.weight.data)
        nn.init.xavier_uniform_(self.rel_transfer.weight.data)

    def _resize(self, tensor, axis, size):
        shape = tensor.size()
        osize = shape[axis]
        if osize == size:
            return tensor
        if (osize > size):
            return torch.narrow(tensor, axis, 0, size)
        paddings = []
        for i in range(len(shape)):
            if i == axis:
                paddings = [0, size - osize] + paddings
            else:
                paddings = [0, 0] + paddings
        print(paddings)
        return F.pad(tensor, paddings=paddings, mode="constant", value=0)

    # 权重预定义
    def input_pre_transd(self, ent_vector, rel_vector, ent_transfer,rel_transfer):
        for i in range(self.entity_num):
            self.ent_embedding.weight.data[i] = torch.from_numpy(np.array(ent_vector[i]))
        for i in range(self.relation_num):
            self.rel_embedding.weight.data[i] = torch.from_numpy(np.array(rel_vector[i]))
        for i in range(self.entity_num):
            self.ent_transfer.weight.data[i] = torch.from_numpy(np.array(ent_transfer[i]))
        for i in range(self.relation_num):
            self.rel_transfer.weight.data[i] = torch.from_numpy(np.array(rel_transfer[i]))

    # 将头实体和尾实体分别映射到不同空间中
    def transfer(self,e,e_transfer,r_transfer):
        if e.shape[0] != r_transfer.shape[0]:
            e = e.view(-1,r_transfer.shape[0],e.shape[-1])
            e_transfer = e_transfer.view(-1, r_transfer.shape[0], e_transfer.shape[-1])
            r_transfer = r_transfer.view(-1, r_transfer.shape[0], r_transfer.shape[-1])
            e = F.normalize(
                self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
                p=2,
                dim=-1
            )
            return e.view(-1, e.shape[-1])
        else:
            return F.normalize(
                self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
                p=2,
                dim=-1
            )

    # 定义的距离函数，同时也是得分函数，衡量一个三元组的正确性
    def distance(self, h, r, t):
        head = self.ent_embedding(h)
        rel = self.rel_embedding(r)
        tail = self.ent_embedding(t)

        h_transfer = self.ent_transfer(h)
        r_transfer = self.rel_transfer(r)
        t_transfer = self.ent_transfer(t)

        head = self.transfer(head,h_transfer,r_transfer)
        tail = self.transfer(tail,t_transfer,r_transfer)

        head = F.normalize(head, 2, -1)
        rel = F.normalize(rel, 2, -1)
        tail = F.normalize(tail, 2, -1)

        distance = head + rel - tail
        score = torch.norm(distance, p=self.norm, dim=1)
        return score

    # 基于上面的distance函数定义的用于测试的打分函数
    def test_distance(self, h, r, t):
        head = self.ent_embedding(h.cuda())
        rel = self.rel_embedding(r.cuda())
        tail = self.ent_embedding(t.cuda())

        h_transfer = self.ent_transfer(h.cuda())
        r_transfer = self.rel_transfer(r.cuda())
        t_transfer = self.ent_transfer(t.cuda())

        head = self.transfer(head, h_transfer, r_transfer)
        tail = self.transfer(tail, t_transfer, r_transfer)
        distance = head + rel - tail
        score = torch.norm(distance, p=self.norm, dim=1)
        return score.cpu().detach().numpy()

    def scale_loss(self, embedding):
        return torch.sum(
            torch.max(
                torch.sum(
                    embedding ** 2, dim=1, keepdim=True
                )-torch.autograd.Variable(torch.FloatTensor([1.0]).to(device)),
                torch.autograd.Variable(torch.FloatTensor([0.0]).to(device))
            ))

    # 计算过程
    def forward(self, current_triples, corrupted_triples):
        h, r, t = torch.chunk(current_triples, 3, dim=1)
        h_c, r_c, t_c = torch.chunk(corrupted_triples, 3, dim=1)
        # 删除tensor中值为1的维度
        h = torch.squeeze(h, dim=1).to(device)
        r = torch.squeeze(r, dim=1).to(device)
        t = torch.squeeze(t, dim=1).to(device)
        h_c = torch.squeeze(h_c, dim=1).to(device)
        r_c = torch.squeeze(r_c, dim=1).to(device)
        t_c = torch.squeeze(t_c, dim=1).to(device)
        # torch.nn.embedding类的forward只接受longTensor类型的张量
        pos = self.distance(h, r, t)
        neg = self.distance(h_c, r_c, t_c)
        entity_embedding = self.ent_embedding(torch.cat([h, t, h_c, t_c]).to(device))
        relation_embedding = self.rel_embedding(torch.cat([r, r_c]).to(device))
        y = Variable(torch.Tensor([-1])).to(device)
        loss = self.loss_F(pos, neg, y)
        ent_scale_loss = self.scale_loss(entity_embedding)
        rel_scale_loss = self.scale_loss(relation_embedding)
        return loss


# TransD主类
class TransD:
    def __init__(self, entity, relation, triple_list, embedding_dim, lr, margin=1.0, norm=1, C = 1.0, valid_triple_list = None):
        self.entities = entity
        self.relations = relation
        self.triples = triple_list
        self.dimension = embedding_dim
        self.learning_rate = lr
        self.margin = margin
        self.norm = norm
        self.loss = 0.0
        self.valid_loss = 0.0
        self.valid_triples = valid_triple_list
        self.train_loss = []
        self.validation_loss = []
        self.test_triples = []
        self.C = C

    def data_initialise(self):
        self.model = TransD_Model(len(self.entities), len(self.relations), self.dimension, self.margin, self.norm, self.C)
        self.optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def insert_pre_data(self, file1, file2, file3, file4):
        entity_dic = {}
        relation = {}
        rel_mat = {}
        ent_mat = {}
        with codecs.open(file1, 'r') as f1, codecs.open(file2, 'r') as f2, codecs.open(file3, 'r') as f3, codecs.open(file4,'r') as f4:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            lines3 = f3.readlines()
            lines4 = f4.readlines()
            for line in lines1:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                entity_dic[int(line[0])] = json.loads(line[1])
            for line in lines2:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                relation[int(line[0])] = json.loads(line[1])
            for line in lines3:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                rel_mat[int(line[0])] = json.loads(line[1])
            for line in lines4:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                ent_mat[int(line[0])] = json.loads(line[1])
        self.model.input_pre_transd(entity_dic, relation, ent_mat,rel_mat)

    def insert_test_data(self, file1, file2, file3, file4, file5):
        self.insert_pre_data(file1, file2,file3,file4)
        triple_list = []
        with codecs.open(file5, 'r') as f:
          content = f.readlines()
          for line in content:
              triple = line.strip().split("\t")
              if len(triple) != 3:
                  continue
              head = int(entities2id[triple[0]])
              relation = int(relations2id[triple[1]])
              tail = int(entities2id[triple[2]])
              triple_list.append([head, relation, tail])
        self.test_triples = triple_list

    def insert_traning_data(self, file1, file2, file3):
        self.insert_pre_data(file1, file2)
        with codecs.open(file3, 'r') as f5:
            lines = f5.readlines()
            for line in lines:
                line = line.strip().split('\t')
                self.train_loss = json.loads(line[0])
                self.validation_loss = json.loads(line[1])
        print(self.train_loss, self.validation_loss)

    def training_run(self, epochs, batch_size, out_file_title):
        df = pd.DataFrame(columns=["epoch", "time_cost", "train_loss", "valid_loss"])
        n_batches = int(len(self.triples) / batch_size)
        valid_batch = int(len(self.valid_triples) / batch_size) + 1
        print("batch size: ", n_batches, "valid_batch: " , valid_batch)
        # 迭代并且进行参数更新
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0.0
            self.valid_loss = 0.0
            # Normalise the embedding of the entities to 1
            for batch in range(n_batches):
                batch_samples = random.sample(self.triples, batch_size)
                current = []
                corrupted = []
                for sample in batch_samples:
                    corrupted_sample = copy.deepcopy(sample)
                    pr = np.random.random(1)[0]
                    p = relation_tph[int(corrupted_sample[1])] / (relation_tph[int(corrupted_sample[1])] + relation_hpt[int(corrupted_sample[1])])
                    '''
                    tph -> 表示每一个头实体对应的平均尾实体数 
                    hpt -> 表示每一个尾实体对应的平均头实体数
                    定义这两个参数的目的就在于更好的决定替换头实体或是尾实体，从而生成更好的用于训练的corrupted triples负样本
                    因此，替换头实体或尾实体的原则，
                        当 tph > hpt 时，此时一个头实体对应多个尾实体，因此更倾向于替换头实体 
                        当 hpt > tph 时，此时一个尾实体对应多个头实体，因此更倾向于替换尾实体
                    '''
                    if pr < p:
                        # 随机替换头实体，构造负样本三元组
                        corrupted_sample[0] = random.sample(self.entities, 1)[0]
                        while corrupted_sample[0] == sample[0]:
                            corrupted_sample[0] = random.sample(self.entities, 1)[0]
                    else:
                        # 随机替换尾实体，构造负样本三元组
                        corrupted_sample[2] = random.sample(self.entities, 1)[0]
                        while corrupted_sample[2] == sample[2]:
                            corrupted_sample[2] = random.sample(self.entities, 1)[0]

                    current.append(sample)
                    corrupted.append(corrupted_sample)
                current = torch.from_numpy(np.array(current)).long()
                corrupted = torch.from_numpy(np.array(corrupted)).long()
                self.update_triple_embedding(current, corrupted)
            # 验证集上验证结果
            for batch in range(valid_batch):
                batch_samples = random.sample(self.valid_triples, batch_size)
                current = []
                corrupted = []
                for sample in batch_samples:
                    corrupted_sample = copy.deepcopy(sample)
                    pr = np.random.random(1)[0]
                    p = relation_tph[int(corrupted_sample[1])] / (
                            relation_tph[int(corrupted_sample[1])] + relation_hpt[int(corrupted_sample[1])])
                    if pr > p:
                        # change the head entity
                        corrupted_sample[0] = random.sample(self.entities, 1)[0]
                        while corrupted_sample[0] == sample[0]:
                            corrupted_sample[0] = random.sample(self.entities, 1)[0]
                    else:
                        # change the tail entity
                        corrupted_sample[2] = random.sample(self.entities, 1)[0]
                        while corrupted_sample[2] == sample[2]:
                            corrupted_sample[2] = random.sample(self.entities, 1)[0]

                    current.append(sample)
                    corrupted.append(corrupted_sample)

                current = torch.from_numpy(np.array(current)).long()
                corrupted = torch.from_numpy(np.array(corrupted)).long()
                self.calculate_valid_loss(current, corrupted)
            end = time.time()
            mean_train_loss = self.loss / n_batches
            mean_valid_loss = self.valid_loss / valid_batch
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("Train loss: ", mean_train_loss, "Valid loss: ", mean_valid_loss)
            df.loc[epoch] = [epoch, round((end - start), 3), float(mean_train_loss), float(mean_valid_loss)]
            self.train_loss.append(float(mean_train_loss))
            self.validation_loss.append(float(mean_valid_loss))

        # 训练过程中的损失函数可视化代码  code for visulization
        fig = plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(self.train_loss) + 1), self.train_loss, label='Train Loss')
        plt.plot(range(1, len(self.validation_loss) + 1), self.validation_loss, label='Validation Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.xlim(0, len(self.train_loss) + 1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.title(out_file_title + "TransD Training loss")
        plt.show()
        # 保存在训练集上和验证集上的损失函数变化曲线
        fig.savefig(out_file_title + 'TransD_loss_plot.png', bbox_inches='tight')

        # 写入entity的embedding结果
        with codecs.open(out_file_title +"TransD_entity_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f1:
            for i, e in enumerate(self.model.ent_embedding.weight):
                f1.write(str(i) + "\t")
                f1.write(str(e.cpu().detach().numpy().tolist()))
                f1.write("\n")

        # 写入relation的embedding结果
        with codecs.open(out_file_title +"TransD_relation_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f2:
            for i, e in enumerate(self.model.rel_embedding.weight):
                f2.write(str(i) + "\t")
                f2.write(str(e.cpu().detach().numpy().tolist()))
                f2.write("\n")

        with codecs.open(out_file_title+"TransD_ent_transfer_"+ str(self.dimension) + "_"+ str(self.dimension) +"dim_batch" + str(batch_size), "w") as f3:
            for i, e in enumerate(self.model.ent_transfer.weight):
                f3.write(str(i) + "\t")
                f3.write(str(e.cpu().detach().numpy().tolist()))
                f3.write("\n")

        with codecs.open(out_file_title+"TransD_rel_transfer_"+ str(self.dimension) + "_"+ str(self.dimension) +"dim_batch" + str(batch_size), "w") as f4:
            for i, e in enumerate(self.model.rel_transfer.weight):
                f4.write(str(i) + "\t")
                f4.write(str(e.cpu().detach().numpy().tolist()))
                f4.write("\n")
        # 记录损失函数的变化情况
        with codecs.open(out_file_title + "TransD_loss_record.txt", "w") as f5:
                f5.write(str(self.train_loss)+ "\t" + str(self.validation_loss))

        return df

    # 利用torch的梯度回传api和参数更新step来优化
    def update_triple_embedding(self, correct_sample, corrupted_sample):
        self.optim.zero_grad()
        loss = self.model(correct_sample, corrupted_sample)
        self.loss += loss
        loss.backward()
        self.optim.step()

    def calculate_valid_loss(self, correct_sample, corrupted_sample):
        loss = self.model(correct_sample, corrupted_sample)
        self.valid_loss += loss

    def test_run(self, filter=False):
        self.filter = filter
        hits = 0
        rank_sum = 0
        num = 0

        for triple in self.test_triples:
            start = time.time()
            num += 1
            print(num, triple)
            rank_head_dict = {}
            rank_tail_dict = {}
            head_embedding = []
            tail_embedding = []
            norm_relation = []
            hyper_relation = []
            tamp = []
            head_filter = []
            tail_filter = []
            if self.filter:
                for tr in self.triples:
                    if tr[1] == triple[1] and tr[2] == triple[2] and tr[0] != triple[0]:
                        head_filter.append(tr)
                    if tr[0] == triple[0] and tr[1] == triple[1] and tr[2] != triple[2]:
                        tail_filter.append(tr)
                for tr in self.test_triples:
                    if tr[1] == triple[1] and tr[2] == triple[2] and tr[0] != triple[0]:
                        head_filter.append(tr)
                    if tr[0] == triple[0] and tr[1] == triple[1] and tr[2] != triple[2]:
                        tail_filter.append(tr)
                for tr in self.valid_triples:
                    if tr[1] == triple[1] and tr[2] == triple[2] and tr[0] != triple[0]:
                        head_filter.append(tr)
                    if tr[0] == triple[0] and tr[1] == triple[1] and tr[2] != triple[2]:
                        tail_filter.append(tr)

            for i, entity in enumerate(self.entities):
                head_triple = [entity, triple[1], triple[2]]
                if self.filter:
                    if head_triple in head_filter:
                        continue
                head_embedding.append(head_triple[0])
                norm_relation.append(head_triple[1])
                tail_embedding.append(head_triple[2])
                tamp.append(tuple(head_triple))

            head_embedding = torch.from_numpy(np.array(head_embedding)).long()
            norm_relation = torch.from_numpy(np.array(norm_relation)).long()
            tail_embedding = torch.from_numpy(np.array(tail_embedding)).long()
            distance = self.model.test_distance(head_embedding, norm_relation, tail_embedding)  # 打分函数

            for i in range(len(tamp)):
                rank_head_dict[tamp[i]] = distance[i]
            head_embedding = []
            tail_embedding = []
            norm_relation = []
            tamp = []
            for i, tail in enumerate(self.entities):
                tail_triple = [triple[0], triple[1], tail]
                if self.filter:
                    if tail_triple in tail_filter:
                        continue
                head_embedding.append(tail_triple[0])
                norm_relation.append(tail_triple[1])
                tail_embedding.append(tail_triple[2])
                tamp.append(tuple(tail_triple))

            head_embedding = torch.from_numpy(np.array(head_embedding)).long()
            norm_relation = torch.from_numpy(np.array(norm_relation)).long()
            tail_embedding = torch.from_numpy(np.array(tail_embedding)).long()
            distance = self.model.test_distance(head_embedding, norm_relation, tail_embedding)
            for i in range(len(tamp)):
                rank_tail_dict[tamp[i]] = distance[i]
            rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1), reverse=False)
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1), reverse=False)
            # 计算mean_rank和hit_10
            i = 0
            for i in range(len(rank_head_sorted)):
                if triple[0] == rank_head_sorted[i][0][0]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break
            # tail rank
            i = 0
            for i in range(len(rank_tail_sorted)):
                if triple[2] == rank_tail_sorted[i][0][2]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break
            end = time.time()
            print("epoch: ", num, "cost time: %s" % (round((end - start), 3)), str(hits / (2 * num)),
                  str(rank_sum / (2 * num)))
        self.hit_10 = hits / (2 * len(self.test_triples))
        self.mean_rank = rank_sum / (2 * len(self.test_triples))
        # 输出测试集上的测试结果，结果展示的第三部分
        return self.hit_10, self.mean_rank


# main 函数入口
# 初始在单个py文件中放入所有代码逻辑
# 后续考虑将 ”数据读取“、”模型“、”训练函数“、”测试函数“、”可视化“ 进行解耦
if __name__ == '__main__':

    # # 读入的数据集
    # """
    #     file1：训练集三元组
    #     file2：entity2id文件
    #     file3：relation2id文件
    #     file4：验证集三元组
    # """
    # # file1 = "WN18\\wordnet-mlj12-train.txt"
    # # file2 = "WN18\\entity2id.txt"
    # # file3 = "WN18\\relation2id.txt"
    # # file4 = "WN18\\wordnet-mlj12-valid.txt"
    # file1 = r"D:\Codes\网安综合实验\datasets\WN18\wordnet-mlj12-train.txt"
    # file2 = r"D:\Codes\网安综合实验\datasets\WN18\entity2id.txt"
    # file3 = r"D:\Codes\网安综合实验\datasets\WN18\relation2id.txt"
    # file4 = r"D:\Codes\网安综合实验\datasets\WN18\wordnet-mlj12-valid.txt"
    # entity_set, relation_set, triple_list, valid_triple_list = dataloader(file1, file2, file3, file4)
    #
    # # 写入的实体和关系的embedding结果
    # """
    #     file5：实体embedding结果文件
    #     file6：关系embedding结果文件
    # """
    # file5 = "WN18_torch_TransE_entity_50dim_batch4800"
    # file6 = "WN18_torch_TransE_relation_50dim_batch4800"
    #
    # # 在测试集上测试训练的模型的效果
    # """
    #     file8：测试集三元组文件
    # """
    # file8 = "WN18\\wordnet-mlj12-test.txt"
    # # file9 = "Fb15k_loss_record.txt"
    # transE = TransE(entity_set, relation_set, triple_list, embedding_dim=50, lr=0.01, margin=6.0, norm=1, C = 0.25, valid_triple_list=valid_triple_list)
    # transE.data_initialise()
    # df = transE.training_run(epochs=50, batch_size=4800,out_file_title="WN18_torch_")
    # df.to_csv("param.csv")
    # # transE.insert_test_data(file5, file6, file8)
    # #
    # # hit_10,meanRank = transE.test_run(filter=True)
    # # print(f"hit@10:{hit_10}\n mean_rank:{meanRank}")


    ## 以下代码为在FB15K上的实验
    file1 = "FB15k\\freebase_mtr100_mte100-train.txt"
    file2 = "FB15k\\entity2id.txt"
    file3 = "FB15k\\relation2id.txt"
    file4 = "FB15k\\freebase_mtr100_mte100-valid.txt"

    file5 = "FB15k_torch_TransD_entity_50dim_batch9600"
    file6 = "FB15k_torch_TransD_relation_50dim_batch9600"

    file7 = "FB15k_torch_TransD_rel_transfer_50_50dim_batch9600"
    file8 = "FB15k_torch_TransD_ent_transfer_50_50dim_batch9600"

    file9 = "FB15k\\freebase_mtr100_mte100-test.txt"
    # file9 = "Fb15k_loss_record.txt"
    entity_set, relation_set, triple_list, valid_triple_list = dataloader(file1, file2, file3, file4)

    transd = TransD(entity_set, relation_set, triple_list, embedding_dim=50, lr=0.01, margin=4.0, norm=1, C = 0.25, valid_triple_list=valid_triple_list)
    transd.data_initialise()
    # transd.training_run(epochs=20, batch_size=9600, out_file_title="FB15k_torch_")
    transd.insert_test_data(file5, file6, file7, file8, file9)
    transd.test_run(filter=False)
