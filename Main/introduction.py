import streamlit as st
from PIL import Image

# 算法展示模块

TransE_intro = """
-   原论文：《Translating Embeddings for Modeling Multi-relational Data》
-   基本思想：将relation看作从头实体head到尾实体tail的翻译，认为正确的三元组嵌入应当满足head+relation≈tail，而错误的三元组则尽可能不满足这一情况
"""

TransH_intro = """
-   原论文：《Knowledge graph embedding by translating on hyperplanes》
-   TransE对于简单关系的建模效果显著，但是对复杂关系例如一对多、多对一、多对多关系的建模效果并不理想，TransH模型对此进行了改进
-   基本思想：对于每一个关系建立一个超平面，将头实体和尾实体分别投影到该超平面上，使得投影之后的嵌入向量满足 h⊥+dr ≈ t⊥；这样就可以使得同一个实体在不同的关系中体现出不同的意义；
"""

TransR_intro = """
-   原论文：《Learning Entity and Relation Embeddings for Knowledge Graph Completion》
-   TransE和TransH算法的实体和关系都是在同一语义空间中，但是实体包含多种属性，不同的关系体现出了实体的不同属性。因此，在实体空间中，有些实体是相似的，因而彼此接近；但某些特定属性却相当不同，因此在相应的关系空间中，这两个实体则应当互相远离，TransR则起源于此。
-   基本思想：TransR将实体和关系映射到不同语义空间，头尾实体的翻译也是在相应的关系空间中完成。对于每一个三元组(h, r, t)，将头尾实体表示在实体空间，将关系表示在关系空间，并且，对于每一个关系 r，存在一个映射矩阵 Wr，通过这个矩阵将 h, t 映射到关系 r 所在空间，得到 hr 和 tr，使 hr + r = tr。在这种关系的作用下，具有这种关系的头/尾实体彼此接近(彩色的圆圈)，不具有此关系（彩色的三角形）的实体彼此远离。
"""

TransD_intro = """
-   原论文：《Knowledge graph embedding via dynamic mapping matrix》
-   TransR也存在一些缺点：
    -   (1) 在同一关系 r 下, 头尾实体共用相同的投影矩阵，然而，一个关系的头尾实体存在很大的差异，例如（美国，总统，奥巴马），美国是一个实体，代表国家，奥巴马是一个实体，代表的是人物。
    -   (2) TransR 仅仅让给投影矩阵与关系有关是不合理的，因为投影矩阵是头尾实体与关系的交互过程，应该与实体和关系都相关。
    -   (3) TransR 模型的参数急剧增加，计算的时间复杂度大大提高。
-   基本思想：给定一个三元组(h, r, t)，TransD 将头尾实体分别投影到关系空间得到投影矩阵 Mrh 和 Mrt ,这样得到的投影矩阵便与实体和关系都有关系。获取投影矩阵之后，和 TransR 一样，计算头尾实体的投影向量。
"""

Freebase15k_intro = """
This's FB15k data set!
The details of it can be get by this paper titled:A. Bordes, N. Usunier, A. Garcia-Duran, J. Weston and O. Yakhnenko. Translating Embeddings for Modeling Multi-relational Data. In Advances of Neural Information Processing Systems 2013. 
"""

Wordnet_intro = """
This's WN18 data set!
The details of it can be get by this paper titled: A. Bordes, X. Glorot, J. Weston and Y. Bengio. A Semantic Matching Energy Function for Learning with Multi-relational Data. Machine Learning Journal - Special Issue on Learning Semantics. 2012.
"""

Countries_intro = """
This's Countries data set!
The details of it can be get by this paper titled:Bouchard, G.; Singh, S.; and Trouillon, T. 2015. On approximate reasoning capabilities of low-rank vector spaces. AAAI Spring Syposium on Knowledge Representation and Reasoning (KRR): Integrating Symbolic and Neural Approaches.
"""

DBpedia50_intro = """
This's DBpedia50 data set!. It can be get by this paper named "Baoxu Shi, Tim Weninger.Open-World Knowledge Graph Completion.AAAI 2018"
"""

WN18RR_intro = """
This's WN18RR data set!
The details of it can be get by this paper titled:Tim Dettmers, Pasquale Minervini,Pontus Stenetorp,Sebastian Riedel. Convolutional 2D Knowledge Graph Embeddings,2018, Association for the Advancement of Artificial Intelligence (www.aaai.org).
"""

UMLS_intro = """
This's UMLS data set!
"""

def alg_introduction(algorithm_name):
    if algorithm_name == "TransE":
        with st.expander("2.TransE算法说明"):
            st.markdown(TransE_intro)
            image = Image.open(r"D:\Codes\网安综合实验\Main\TransE.png")
            st.image(image)
    if algorithm_name == "TransH":
        with st.expander("2.TransH算法说明"):
            st.write(TransH_intro)
            image = Image.open(r"D:\Codes\网安综合实验\Main\TransH.png")
            st.image(image)
    if algorithm_name == "TransR":
        with st.expander("2.TransR算法说明"):
            st.write(TransR_intro)
            image = Image.open(r"D:\Codes\网安综合实验\Main\TransR.png")
            st.image(image)
    if algorithm_name == "TransD":
        with st.expander("2.TransD算法说明"):
            st.write(TransD_intro)
            image = Image.open(r"D:\Codes\网安综合实验\Main\TransD.png")
            st.image(image)

# 数据集展示模块
def dataset_introduction(dataset_name):
    if dataset_name == "FreeBase15K":
        with st.expander("3.FreeBase15K数据集简介"):
            st.write(Freebase15k_intro)
    if dataset_name == "WordNet18":
        with st.expander("3.WordNet18数据集简介"):
            st.write(Wordnet_intro)
    if dataset_name == "Countries":
        with st.expander("3.Countries数据集简介"):
            st.write(Countries_intro)
    if dataset_name == "DBpedia50":
        with st.expander("3.DBpedia50数据集简介"):
            st.write(DBpedia50_intro)
    if dataset_name == "WN18RR":
        with st.expander("3.WN18RR数据集简介"):
            st.write(WN18RR_intro)
    if dataset_name == "UMLS":
        with st.expander("3.UMLS数据集简介"):
            st.write(UMLS_intro)


source_code_E = """
class TransE_Model(nn.Module):
    def __init__(self, entity_num, relation_num, dim, margin, norm, C):
        '''
        :param entity_num:实体数目
        :param relation_num: 关系数目
        :param dim: embedding的维度
        :param margin: 正确三元组和错误的三元组之间的间隔修正，越大对于嵌入词向量的修改就越严格
        :param norm:距离的计算
        :param C:正则化系数
        '''
        super(TransE_Model, self).__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.dim = dim
        self.margin = margin
        self.norm = norm
        self.C = C

        # 实体的初始embedding层
        self.ent_embedding = torch.nn.Embedding(num_embeddings=self.entity_num,embedding_dim=self.dim).to(device)
        # 关系的初始embedding层
        self.rel_embedding = torch.nn.Embedding(num_embeddings=self.relation_num,embedding_dim=self.dim).to(device)
        # 损失函数定义
        self.loss_F = nn.MarginRankingLoss(self.margin, reduction="mean").to(device)
        self.__data_init()

    # 参数初始化
    def __data_init(self):
        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        self.normalization_rel_embedding()
        self.normalization_ent_embedding()

    # 实体embedding的规范化
    def normalization_ent_embedding(self):
        norm = self.ent_embedding.weight.detach().cpu().numpy()
        norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        self.ent_embedding.weight.data.copy_(torch.from_numpy(norm))

    # 关系embedding的规范化
    def normalization_rel_embedding(self):
        norm = self.rel_embedding.weight.detach().cpu().numpy()
        norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        self.rel_embedding.weight.data.copy_(torch.from_numpy(norm))

    # 权重预定义
    def input_pre_transe(self, ent_vector, rel_vector):
        for i in range(self.entity_num):
            self.ent_embedding.weight.data[i] = torch.from_numpy(np.array(ent_vector[i]))
        for i in range(self.relation_num):
            self.rel_embedding.weight.data[i] = torch.from_numpy(np.array(rel_vector[i]))

    # 定义的距离函数，同时也是得分函数，衡量一个三元组的正确性
    def distance(self, h, r, t):
        head = self.ent_embedding(h)
        rel = self.rel_embedding(r)
        tail = self.ent_embedding(t)
        distance = head + rel - tail
        score = torch.norm(distance, p=self.norm, dim=1)
        return score

    # 基于上面的distance函数定义的用于测试的打分函数
    def test_distance(self, h, r, t):
        head = self.ent_embedding(h.to(device))
        rel = self.rel_embedding(r.to(device))
        tail = self.ent_embedding(t.to(device))
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
        return loss + self.C * (ent_scale_loss/len(entity_embedding) + rel_scale_loss/len(relation_embedding))
"""
source_code_H = '''
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
'''
source_code_R = '''
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
'''
source_code_D = '''
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
'''
# 代码展示模块
def alg_sourcecode_show(algorithm_name):
    if algorithm_name == 'TransE':
        with st.expander("4.TransE算法核心代码"):
            st.code(source_code_E,'python')
    if algorithm_name == 'TransH':
        with st.expander("4.TransH算法核心代码"):
            st.code(source_code_H,'python')
    if algorithm_name == 'TransR':
        with st.expander("4.TransR算法核心代码"):
            st.code(source_code_R,'python')
    if algorithm_name == 'TransD':
        with st.expander("4.TransD算法核心代码"):
            st.code(source_code_D,'python')