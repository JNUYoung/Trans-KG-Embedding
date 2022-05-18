"""
在TransE和TransH的假设下，相似的实体在统一实体空间中会非常接近；
但是实际中，每个实体有许多不同的方面，不同的关系则对应于实体的不同方面；

因此，TransR为每个关系定义了独有的关系空间；（关系特定的空间） 因此，可以将实体投影到对应的关系空间中，从而体现出实体在该关系下的语义；
例如：
    apple和huawei，在代表科技公司时，例如(apple,located in,Carlifonia)和(huawei,located in, Shenzhen)，那么apple和huawei的embedding就应该相似；
    但是在apple表示水果时，这两者的embedding就应该呈现出极大的不一致的现象；
    这也反映了”关系特定 (relation-specific)“ 的含义；
"""

import codecs
import numpy as np
import copy
import time
import random
import matplotlib.pyplot as plt
import json
import operator # operator模块输出一系列对应Python内部操作符的函数
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

entities2id = {}
relations2id = {}
relation_tph = {}
relation_hpt = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 实体-实体ID
entities2id = {}
# 关系-关系ID
relations2id = {}
# tails per head : 每个头节点所关联的尾节点数目，相当于反映了该节点的出度
relation_tph = {}
# heads per tail : 每个尾节点所关联的头节点数目，相当于反映了该节点的入度
relation_hpt = {}

# 数据加载函数
def dataloader(file1, file2, file3, file4):
    """
    读取知识图谱数据集并返回TransE模型规定的数据格式
    :param file1: 存储参与训练三元组的txt文件，每行的数据格式为：head_entity  relation    tail_entity
    :param file2: entity2id.txt，每行的数据格式为：entity    id
    :param file3: relation2id.txt 每行的数据格式为: relation    id
    :param file4: 验证集三元组txt文件 每行的数据格式为：head_entity  relation    tail_entity
    :return: entity->list, relation->list, triple_list, valid_triple_list
    """
    print("------start extracting EntityList and RelationList------")
    time1 = time.time()
    entity = []
    relation = []
    with open(file2, 'r') as f1, open(file3, 'r') as f2:
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
    triple_list = []
    relation_head = {}
    relation_tail = {}
    print("------starting extracting TriplesList for training------")
    time3 = time.time()
    with codecs.open(file1, 'r') as f:
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
    with codecs.open(file4, 'r') as f:
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


class TransR_Model(nn.Module):
    def __init__(self, entity_num, relation_num, ent_dim, rel_dim, margin, norm, C):
        """
        :param entity_num: 实体的数目
        :param relation_num: 关系的数目
        :param ent_dim: 实体的embedding维度
        :param rel_dim: 关系的embedding维度
        :param margin: 正确三元组和错误的三元组之间的间隔修正，越大对于嵌入词向量的修改就越严格
        :param norm: 范数选择
        :param C: 正则化系数
        """
        super(TransR_Model, self).__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.margin = margin
        self.norm = norm
        self.C = C

        # 定义实体的embedding矩阵
        self.ent_embedding = torch.nn.Embedding(num_embeddings=self.entity_num,embedding_dim=self.ent_dim).cuda()
        # 定义关系的embedding矩阵
        self.rel_embedding = torch.nn.Embedding(num_embeddings=self.relation_num,embedding_dim=self.rel_dim).cuda()
        # 定义关系特定的映射矩阵，用于将实体从实体空间映射到关系空间
        self.rel_matrix = torch.nn.Embedding(num_embeddings= self.relation_num,embedding_dim=self.ent_dim*self.rel_dim).cuda()
        # 损失函数定义
        self.loss_F = nn.MarginRankingLoss(self.margin, reduction="mean").cuda()
        self.__data_init()

    def __data_init(self):
        # embedding.weight (Tensor) -形状为(num_embeddings, embedding_dim)的嵌入中可学习的权值
        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        identity = torch.zeros(self.ent_dim, self.rel_dim)
        for i in range(min(self.ent_dim, self.rel_dim)):
            identity[i][i] = 1
        identity = identity.view(self.ent_dim * self.rel_dim)
        for i in range(self.relation_num):
            self.rel_matrix.weight.data[i] = identity

    def input_pre_transe(self, ent_vector, rel_vector):
        for i in range(self.entity_num):
            self.ent_embedding.weight.data[i] = torch.from_numpy(np.array(ent_vector[i]))
        for i in range(self.relation_num):
            self.rel_embedding.weight.data[i] = torch.from_numpy(np.array(rel_vector[i]))

    def input_pre_transr(self, ent_vector, rel_vector, rel_matrix):
        for i in range(self.entity_num):
            self.ent_embedding.weight.data[i] = torch.from_numpy(np.array(ent_vector[i]))
        for i in range(self.relation_num):
            self.rel_embedding.weight.data[i] = torch.from_numpy(np.array(rel_vector[i]))
        for i in range(self.relation_num):
            self.rel_matrix.weight.data[i] = torch.from_numpy(np.array(rel_matrix[i]))

    # transfer函数实现从实体空间到关系空间的映射
    def transfer(self, e, rel_mat):
        # view 的作用是重新将一个tensor转化成另一个形状
        # 数据按照行优先的顺序排成一个一维的数据（这里应该是因为要求地址是连续存储的），然后按照参数组合成其他维度的tensor
        # -1 表示根据其他维度来自动计算这一维的数量
        # view 的一个作用是，让新数据和原数据共享内存，因此我们在修改rel_matrix 的数据后，原数据rel_mat也会改变
        e = F.normalize(e, 2, -1)
        rel_matrix = rel_mat.view(-1, self.ent_dim, self.rel_dim)
        e = e.view(-1, 1, self.ent_dim)
        e = torch.matmul(e, rel_matrix)
        return e.view(-1, self.rel_dim)

    def distance(self, h, r, t):
        head = self.ent_embedding(h)
        rel = self.rel_embedding(r)
        rel_mat = self.rel_matrix(r)
        tail = self.ent_embedding(t)
        # 将头实体和尾实体分别进行映射
        head = self.transfer(head, rel_mat)
        tail = self.transfer(tail, rel_mat)
        # 基于映射到特定关系空间中的head、relation和tail计算score
        head = F.normalize(head, 2, -1)
        rel = F.normalize(rel, 2, -1)
        tail = F.normalize(tail, 2, -1)
        distance = head + rel - tail

        score = torch.norm(distance, p = self.norm, dim=1)
        return score

    def test_distance(self, h, r, t):
        head = self.ent_embedding(h.cuda())
        rel = self.rel_embedding(r.cuda())
        rel_mat = self.rel_matrix(r.cuda())
        tail = self.ent_embedding(t.cuda())
        head = self.transfer(head, rel_mat)
        tail = self.transfer(tail, rel_mat)
        distance = head + rel - tail
        # dim = -1表示的是维度的最后一维 比如如果一个张量有3维 那么 dim = 2 = -1， dim = 0 = -3
        score = torch.norm(distance, p=self.norm, dim=1)
        return score.cpu().detach().numpy()

    def scale_loss(self, embedding):
        return torch.sum(
            torch.max(
                torch.sum(
                    embedding ** 2, dim=1, keepdim=True
                )-torch.autograd.Variable(torch.FloatTensor([1.0]).cuda()),
                torch.autograd.Variable(torch.FloatTensor([0.0]).cuda())
            ))

    def forward(self, current_triples, corrupted_triples):
        h, r, t = torch.chunk(current_triples, 3, dim=1)
        h_c, r_c, t_c = torch.chunk(corrupted_triples, 3, dim=1)
        h = torch.squeeze(h, dim=1).cuda()
        r = torch.squeeze(r, dim=1).cuda()
        t = torch.squeeze(t, dim=1).cuda()
        h_c = torch.squeeze(h_c, dim=1).cuda()
        r_c = torch.squeeze(r_c, dim=1).cuda()
        t_c = torch.squeeze(t_c, dim=1).cuda()

        entity_embedding = self.ent_embedding(torch.cat([h, t, h_c, t_c]).cuda())
        relation_embedding = self.rel_embedding(torch.cat([r, r_c]).cuda())

        pos = self.distance(h, r, t)
        neg = self.distance(h_c, r_c, t_c)
        y = Variable(torch.Tensor([-1])).cuda()
        loss = self.loss_F(pos, neg, y)
        return loss


class TransR:
    def __init__(self, entity_set, relation_set, triple_list, ent_dim, rel_dim, lr, margin=1.0, norm=1, C = 1.0, valid_triples = None):
        self.entities = entity_set
        self.relations = relation_set
        self.triples = triple_list
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.learning_rate = lr
        self.margin = margin
        self.norm = norm
        self.loss = 0.0
        self.valid_loss = 0.0
        self.valid_triples = valid_triples
        self.C = C

        self.train_loss = []
        self.validation_loss = []
        self.test_triples = []

    def data_initialise(self, transe_ent = None, transe_rel = None):
        self.model = TransR_Model(len(self.entities), len(self.relations), self.ent_dim, self.rel_dim, self.margin, self.norm, self.C)
        self.optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if transe_ent != None and transe_rel != None:
            entity_dic = {}
            relation_dic = {}
            with codecs.open(transe_ent, 'r') as f1, codecs.open(transe_rel, 'r') as f2:
                lines1 = f1.readlines()
                lines2 = f2.readlines()
                for line in lines1:
                    line = line.strip().split('\t')
                    if len(line) != 2:
                        continue
                    entity_dic[int(line[0])] = json.loads(line[1])
                for line in lines2:
                    line = line.strip().split('\t')
                    if len(line) != 2:
                        continue
                    relation_dic[int(line[0])] = json.loads(line[1])
            self.model.input_pre_transe(entity_dic, relation_dic)

    def insert_pre_data(self, file1, file2, file3):
        entity_dic = {}
        relation = {}
        rel_mat = {}
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
                relation[int(line[0])] = json.loads(line[1])
            for line in lines3:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                rel_mat[int(line[0])] = json.loads(line[1])
        self.model.input_pre_transr(entity_dic, relation, rel_mat)

    def insert_test_data(self, file1, file2, file3, file4):
        self.insert_pre_data(file1, file2, file3)
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

    def insert_traning_data(self, file1, file2, file3, file4):
        self.insert_pre_data(file1, file2, file3)
        with codecs.open(file4, 'r') as f5:
            lines = f5.readlines()
            for line in lines:
                line = line.strip().split('\t')
                self.train_loss = json.loads(line[0])
                self.validation_loss = json.loads(line[1])
        print(self.train_loss, self.validation_loss)

    def training_run(self, epochs=300, batch_size=100, out_file_title = ''):
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
                corrupted = torch.from_numpy(np.array(corrupted)).long()
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

                    '''
                    这里关于p的说明 tph 表示每一个头结对应的平均尾节点数 hpt 表示每一个尾节点对应的平均头结点数
                    当tph > hpt 时 更倾向于替换头 反之则跟倾向于替换尾实体

                    举例说明 
                    在一个知识图谱中，一共有10个实体 和n个关系，如果其中一个关系使两个头实体对应五个尾实体，
                    那么这些头实体的平均 tph为2.5，而这些尾实体的平均 hpt只有0.4，
                    则此时我们更倾向于替换头实体，
                    因为替换头实体才会有更高概率获得正假三元组，如果替换头实体，获得正假三元组的概率为 8/9 而替换尾实体获得正假三元组的概率只有 5/9
                    '''
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

            df.loc[epoch] = [epoch, round((end - start), 3), float(mean_train_loss), float(mean_valid_loss)]
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
        plt.title(out_file_title + "TransR Training loss")
        plt.show()
        fig.savefig(out_file_title + 'TransR_loss_plot.png', bbox_inches='tight')
        # .detach()的作用就是返回一个新的tensor，和原来tensor共享内存，但是这个张量会从计算途中分离出来，并且requires_grad=false
        # 由于 能被grad的tensor不能直接使用.numpy(), 所以要是用。detach().numpy()
        with codecs.open(out_file_title+"TransR_entity_" + str(self.rel_dim) + "dim_batch" + str(batch_size), "w") as f1:
            for i, e in enumerate(self.model.ent_embedding.weight):
                f1.write(str(i) + "\t")
                f1.write(str(e.cpu().detach().numpy().tolist()))
                f1.write("\n")

        with codecs.open(out_file_title+"TransR_relation_" + str(self.rel_dim) + "dim_batch" + str(batch_size), "w") as f2:
            for i, e in enumerate(self.model.rel_embedding.weight):
                f2.write(str(i) + "\t")
                f2.write(str(e.cpu().detach().numpy().tolist()))
                f2.write("\n")

        with codecs.open(out_file_title+"TransR_rel_matrix_"+ str(self.ent_dim) + "_"+ str(self.rel_dim) +"dim_batch" + str(batch_size), "w") as f3:
            for i, e in enumerate(self.model.rel_matrix.weight):
                f3.write(str(i) + "\t")
                f3.write(str(e.cpu().detach().numpy().tolist()))
                f3.write("\n")

        with codecs.open(out_file_title + "TransR_loss_record.txt", "w") as f1:
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
            distance = self.model.test_distance(head_embedding, norm_relation, tail_embedding)

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

            distance = self.model.test_distance(head_embedding, norm_relation, tail_embedding)
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

    # file5 = "WN18_torch_TransE_entity_50dim_batch4800"
    # file6 = "WN18_torch_TransE_relation_50dim_batch4800"

    #
    # transR = TransR_Training(entity_set, relation_set, triple_list, ent_dim=50, rel_dim = 50, lr=0.001, margin=4.0, norm=1, C=1.0, valid_triples=valid_triple_list)
    # transR.data_initialise("transE_entity_vector_50dim", "transE_relation_vector_50dim")
    # # transR.data_initialise()
    # transR.training_run(epochs=5, batch_size=1440, out_file_title="WN18_torch_")

    file5 = "FB15k_torch_TransE_entity_50dim_batch9600"
    file6 = "FB15k_torch_TransE_relation_50dim_batch9600"
    #
    transR = TransR(entity_set, relation_set, triple_list, ent_dim=50, rel_dim=50, lr=0.001, margin=6.0,
                             norm=1, C=0.25,  valid_triples=valid_triple_list)
    # transR.data_initialise()
    transR.data_initialise(file5, file6)
    transR.training_run(epochs=100, batch_size=4800, out_file_title="FB15k_torch_")

    # file7 = "FB15k_1torch_TransR_pytorch_entity_50dim_batch4800"
    # file8 = "FB15k_1torch_TransR_pytorch_reltion_50dim_batch4800"
    # file9 = "FB15k_1torch_TransR_pytorch_rel_matrix_50_50dim_batch4800"
    # file10 = "FB15k\\freebase_mtr100_mte100-test.txt"
    # transR = TransR_Training(entity_set, relation_set, triple_list, ent_dim=50, rel_dim=50, lr=0.00001, margin=6.0,
    #                          norm=1, C=0.25, valid_triples=valid_triple_list)
    # transR.data_initialise()
    # transR.insert_test_data(file7, file8, file9, file10)
    # transR.test_run(filter = True)
    # 关于叶节点的说明， 整个计算图中，只有叶节点的变量才能进行自动微分得到梯度，任何变量进行运算操作后，再把值付给他自己，这个变量就不是叶节点了，就不能进行自动微分







