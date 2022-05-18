"""
1.目的：
    为了解决TransE模型的局限性，使得模型能够更好地处理一对多/多对一/多对多关系；
2.基本思想：
    relation → 超平面 ： 每个关系对应一个超平面，将头实体和尾实体映射到该超平面上；
    因此，每个relation有两个向量：一个norm向量(norm_embedding)，一个在该超平面上的平移向量(hyper_embedding)
3.和TransE主要区别：
    需要计算投影向量；
"""


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
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import operator # operator模块输出一系列对应Python内部操作符的函数

entities2id = {}
relations2id = {}
relation_tph = {}
relation_hpt = {}

# 数据加载函数
def dataloader(trainingset, entityset, relationset, validset):
    """
    读取知识图谱数据集并返回TransE模型规定的数据格式
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

# TransH模型基类型定义
class TransH_Model(nn.Module):
    def __init__(self, entity_num, relation_num, dimension, margin, C, epsilon, norm):
        """
        :param entity_num: 实体数目
        :param relation_num: 关系数目
        :param dimension: embedding维度
        :param margin: 正确三元组和错误的三元组之间的间隔修正，越大对于嵌入词向量的修改就越严格
        :param C: 正则化系数
        :param epsilon: 超参数
        :param norm: 范数选择
        """
        super(TransH_Model, self).__init__()
        self.dimension = dimension
        self.margin = margin
        self.C = C
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.epsilon = epsilon
        self.norm = norm

        # 定义实体的embedding矩阵
        # 定义每个关系对应的超平面的法向量矩阵
        # 定义关系在该超平面中的映射矩阵
        self.relation_norm_embedding = torch.nn.Embedding(num_embeddings=relation_num,embedding_dim=self.dimension).cuda()
        self.relation_hyper_embedding = torch.nn.Embedding(num_embeddings=relation_num,embedding_dim=self.dimension).cuda()
        self.entity_embedding = torch.nn.Embedding(num_embeddings=entity_num,embedding_dim=self.dimension).cuda()
        self.loss_F = nn.MarginRankingLoss(self.margin, reduction="mean").cuda()
        self.__data_init()

    # 参数xavier初始化
    def __data_init(self):
        nn.init.xavier_uniform_(self.relation_norm_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_hyper_embedding.weight.data)
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)

    # 权重参数预定义
    def input_pre_transh(self, ent_vector, rel_vector, rel_norm):
        for i in range(self.entity_num):
            self.entity_embedding.weight.data[i] = torch.from_numpy(np.array(ent_vector[i]))
        for i in range(self.relation_num):
            self.relation_hyper_embedding.weight.data[i] = torch.from_numpy(np.array(rel_vector[i]))
        for i in range(self.relation_num):
            self.relation_norm_embedding.weight.data[i] = torch.from_numpy(np.array(rel_norm[i]))

    # 头实体和尾实体基于关系特定超平面法向量的映射函数
    def projected(self, ent, norm):
        norm = F.normalize(norm, p=2, dim=-1)
        return ent - torch.sum(ent * norm, dim = 1, keepdim=True) * norm

    # 将头实体和尾实体映射到关系特定的超平面后，采用与TransE类似的距离度量函数进行评分
    def distance(self, h, r, t):
        head = self.entity_embedding(h)
        r_norm = self.relation_norm_embedding(r)
        r_hyper = self.relation_hyper_embedding(r)
        tail = self.entity_embedding(t)

        head_hyper = self.projected(head, r_norm)
        tail_hyper = self.projected(tail, r_norm)

        distance = head_hyper + r_hyper - tail_hyper
        score = torch.norm(distance, p = self.norm, dim=1)
        return score

    def test_distance(self, h, r, t):
        head = self.entity_embedding(h.cuda())
        r_norm = self.relation_norm_embedding(r.cuda())
        r_hyper = self.relation_hyper_embedding(r.cuda())
        tail = self.entity_embedding(t.cuda())

        head_hyper = self.projected(head, r_norm)
        tail_hyper = self.projected(tail, r_norm)

        distance = head_hyper + r_hyper - tail_hyper
        score = torch.norm(distance, p = self.norm, dim=1)
        return score.cpu().detach().numpy()

    def scale_loss(self, embedding):
        return torch.sum(
            torch.max(
                torch.sum(
                    embedding ** 2, dim=1, keepdim=True
                )-torch.autograd.Variable(torch.FloatTensor([1.0]).cuda()),
                torch.autograd.Variable(torch.FloatTensor([0.0]).cuda())
            ))

    # 正交损失函数
    def orthogonal_loss(self, relation_embedding, w_embedding):
        dot = torch.sum(relation_embedding * w_embedding, dim=1, keepdim=False) ** 2
        norm = torch.norm(relation_embedding, p=self.norm, dim=1) ** 2
        loss = torch.sum(torch.relu(dot / norm - torch.autograd.Variable(torch.FloatTensor([self.epsilon]).cuda() ** 2)))
        return loss

    # 对batch中的正确三元组和错误三元组计算score，返回得到的损失函数值
    def forward(self, current_triples, corrupted_triples):
        h, r, t = torch.chunk(current_triples, 3, dim=1)
        h_c, r_c, t_c = torch.chunk(corrupted_triples, 3, dim=1)

        h = torch.squeeze(h, dim=1).cuda()
        r = torch.squeeze(r, dim=1).cuda()
        t = torch.squeeze(t, dim=1).cuda()
        h_c = torch.squeeze(h_c, dim=1).cuda()
        r_c = torch.squeeze(r_c, dim=1).cuda()
        t_c = torch.squeeze(t_c, dim=1).cuda()

        pos = self.distance(h, r, t)
        neg = self.distance(h_c, r_c, t_c)

        y = Variable(torch.Tensor([-1])).cuda()
        loss = self.loss_F(pos, neg, y)

        entity_embedding = self.entity_embedding(torch.cat([h, t, h_c, t_c]).cuda())
        relation_embedding = self.relation_hyper_embedding(torch.cat([r, r_c]).cuda())
        w_embedding = self.relation_norm_embedding(torch.cat([r, r_c]).cuda())

        orthogonal_loss = self.orthogonal_loss(relation_embedding, w_embedding)
        scale_loss = self.scale_loss(entity_embedding)
        # 软约束项的建议 模长约束对结果收敛有影响，但是正交约束影响很小.所以模长约束保留，正交约束可以不加
        return loss + self.C * (scale_loss/len(entity_embedding) + orthogonal_loss/len(relation_embedding))

class TransH:
    def __init__(self, entity_set, relation_set, triple_list, embedding_dim, lr, margin=1.0, norm=1, C=1.0, epsilon = 1e-5, valid_triple_list = None):
        self.entities = entity_set
        self.relations = relation_set
        self.triples = triple_list
        self.dimension = embedding_dim
        self.learning_rate = lr
        self.margin = margin
        self.norm = norm
        self.loss = 0.0
        self.entity_embedding = {}
        self.norm_relations = {}
        self.hyper_relations = {}
        self.C = C
        self.epsilon = epsilon
        self.valid_triples = valid_triple_list
        self.valid_loss = 0

        self.test_triples = []
        self.train_loss = []
        self.validation_loss = []


    def data_initialise(self):
        self.model = TransH_Model(len(self.entities), len(self.relations), self.dimension, self.margin, self.C, self.epsilon, self.norm)
        self.optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.optim = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def insert_data(self, file1, file2, file3, file4, file5):
        entity_dic = {}
        norm_relation = {}
        hyper_relation = {}
        with codecs.open(file1, 'r') as f1, codecs.open(file2, 'r') as f2, codecs.open(file3, 'r') as f3:
          lines1 = f1.readlines()
          lines2 = f2.readlines()
          lines3 = f3.readlines()
          for line in lines1:
              line = line.strip().split('\t')
              if len(line) != 2:
                  continue
              entity_dic[int(line[0])] = json.loads(line[1])

          for line in lines2:
              line = line.strip().split('\t')
              if len(line) != 2:
                  continue
              norm_relation[int(line[0])] = json.loads(line[1])

          for line in lines3:
              line = line.strip().split('\t')
              if len(line) != 2:
                  continue
              hyper_relation[int(line[0])] = json.loads(line[1])
        self.model.input_pre_transh(entity_dic, hyper_relation, norm_relation)
        # 插入测试数据
        triple_list = []
        with codecs.open(file4, 'r') as f4:
          content = f4.readlines()
          for line in content:
              triple = line.strip().split("\t")
              if len(triple) != 3:
                  continue

              head = int(entities2id[triple[0]])
              relation = int(relations2id[triple[1]])
              tail = int(entities2id[triple[2]])
              triple_list.append([head, relation, tail])

        self.test_triples = triple_list
        # insert_training_data
        with codecs.open(file5, 'r') as f5:
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
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0.0
            self.valid_loss = 0.0
            for batch in range(n_batches):
                batch_samples = random.sample(self.triples, batch_size)
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
                corrupted =  torch.from_numpy(np.array(corrupted)).long()
                self.update_triple_embedding(current, corrupted)

            for batch in range(valid_batch):
                batch_samples = random.sample(self.valid_triples, batch_size)
                current = []
                corrupted = []
                for sample in batch_samples:
                    corrupted_sample = copy.deepcopy(sample)
                    pr = np.random.random(1)[0]
                    p = relation_tph[int(corrupted_sample[1])] / (
                            relation_tph[int(corrupted_sample[1])] + relation_hpt[int(corrupted_sample[1])])
                    if pr < p:
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
            df.loc[epoch] = [epoch,round((end-start),3),float(mean_train_loss),float(mean_valid_loss)]
            self.train_loss.append(float(mean_train_loss))
            self.validation_loss.append(float(mean_valid_loss))

        # visualize the loss as the network trained
        fig = plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(self.train_loss) + 1), self.train_loss, label='Train Loss')
        plt.plot(range(1, len(self.validation_loss) + 1), self.validation_loss, label='Validation Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.xlim(0, len(self.train_loss) + 1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.title(out_file_title + "TransH Training loss")
        plt.show()
        fig.savefig(out_file_title+'TransH_loss_plot.png', bbox_inches='tight')

        with codecs.open(out_file_title + "TransH_entity_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f1:
            for i, e in enumerate(self.model.entity_embedding.weight):
                f1.write(str(i) + "\t")
                f1.write(str(e.cpu().detach().numpy().tolist()))
                f1.write("\n")

        with codecs.open(out_file_title + "TransH_norm_relations_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f2:
            for i, e in enumerate(self.model.relation_norm_embedding.weight):
                f2.write(str(i) + "\t")
                f2.write(str(e.cpu().detach().numpy().tolist()))
                f2.write("\n")

        with codecs.open(out_file_title + "TransH_hyper_relations_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f3:
            for i, e in enumerate(self.model.relation_hyper_embedding.weight):
                f3.write(str(i) + "\t")
                f3.write(str(e.cpu().detach().numpy().tolist()))
                f3.write("\n")

        with codecs.open(out_file_title + "loss_record.txt", "w") as f1:
                f1.write(str(self.train_loss)+ "\t" + str(self.validation_loss))

        return df

    def update_triple_embedding(self, correct_sample, corrupted_sample):
        self.optim.zero_grad()
        loss = self.model(correct_sample, corrupted_sample)
        self.loss += loss
        loss.backward()
        self.optim.step()

    def calculate_valid_loss(self, correct_sample, corrupted_sample):
        loss = self.model(correct_sample, corrupted_sample)
        self.valid_loss += loss

    def test_run(self, filter = False):
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
            distance  = self.model.test_distance(head_embedding, norm_relation, tail_embedding)

            for i in range(len(tamp)):
                rank_head_dict[tamp[i]] = distance[i]

            head_embedding = []
            tail_embedding = []
            norm_relation = []
            hyper_relation = []
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

            distance  = self.model.test_distance(head_embedding, norm_relation, tail_embedding)
            for i in range(len(tamp)):
                rank_tail_dict[tamp[i]] = distance[i]
            rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1), reverse=False)
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1), reverse=False)
            # calculate the mean_rank and hit_10
            # head data
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

        return self.hit_10, self.mean_rank



if __name__ == '__main__':
    # file1 = "WN18\\wordnet-mlj12-train.txt"
    # file2 = "WN18\\entity2id.txt"
    # file3 = "WN18\\relation2id.txt"
    # file4 = "WN18\\wordnet-mlj12-valid.txt"

    file1 = "FB15k\\freebase_mtr100_mte100-train.txt"
    file2 = "FB15k\\entity2id.txt"
    file3 = "FB15k\\relation2id.txt"
    file4 = "FB15k\\freebase_mtr100_mte100-valid.txt"
    entity_set, relation_set, triple_list, valid_triple_list = dataloader(file1, file2, file3, file4)

    # file5 = ""
    # file6 = ""
    # file7 = ""

    # file5 = "WN18_1epoch_TransH_pytorch_entity_50dim_batch4800"
    # file6 = "WN18_1epoch_TransH_pytorch_norm_relations_50dim_batch4800"
    # file7 = "WN18_1epoch_TransH_pytorch_hyper_relations_50dim_batch4800"
    # file8 = "WN18\\wordnet-mlj12-test.txt"
    # file9 = "Fb15k_loss_record.txt"
    # transH = TransH(entity_set, relation_set, triple_list, embedding_dim=50, lr=0.005, margin=4.0, norm=1, C=0.25, epsilon=1e-5, valid_triple_list = valid_triple_list)
    # transH.data_initialise()
    # transH.insert_data(file5, file6, file7, file8, file9)
    # # transH.training_run(epochs=500, batch_size=4800, out_file_title="WN18_1epoch_")
    # transH.test_run(filter = False)


    file5 = "FB15k_50epoch_TransH_pytorch_entity_200dim_batch1200"
    file6 = "FB15k_50epoch_TransH_pytorch_norm_relations_200dim_batch1200"
    file7 = "FB15k_50epoch_TransH_pytorch_hyper_relations_200dim_batch1200"
    file8 = "FB15k\\freebase_mtr100_mte100-test.txt"
    file9 = "Fb15k_loss_record.txt"
    transH = TransH(entity_set, relation_set, triple_list, embedding_dim=200, lr=0.01, margin=8.0, norm=1, C=1.0, epsilon=1e-5, valid_triple_list = valid_triple_list)
    transH.data_initialise()
    # transH.insert_data(file5, file6, file7, file8, file9)
    transH.training_run(epochs=5, batch_size=4800, out_file_title="FB15k_epoch_")
    # transH.test_run(filter = False)








