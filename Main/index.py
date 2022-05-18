# from email.mime import image
import streamlit as st
import time
import numpy as np
import pandas as pd
from PIL import Image
import codecs
import os.path as osp
from zipfile import ZipFile
from io import BytesIO

import sys
sys.path.append(osp.join(osp.dirname(__file__), '..'))
from TransE import TransE_pytorch
from TransR import TransR_pytorch
from TransH import TransH_pytorch
from TransD import TransD_pytorch
from utils import *
from introduction import *

# global variable
# 训练用的文件
trainTriples = None
entityToId = None
relationToId = None
validTriples = None

# 实体和关系embedding结果写入的文件路径
entityEmbRes = None
relationEmbRes = None

# 测试阶段用的文件
testTriples = None

# 结果展示用的文件
csv_filePath = None     # TransE
fig_path = None         # 损失函数图片

raw_hit_10 = 0
raw_mean_rank = 0
filtered_hit_10 = 0
filtered_mean_rank = 0

# 一、参数变化情况展示函数
def params_show(file):
    st.write("一、算法训练过程中的参数展示：")
    _df = pd.read_csv(csv_filePath, index_col=0)
    st.dataframe(data=_df, width=1000, height=400)

# 二、损失函数变化情况展示函数
def loss_plot(file):
    with st.container():
        st.write("二、测试集和训练集的损失函数变化情况：")
        image = Image.open(fig_path)
        st.image(image, caption="损失函数变化情况")

# 三、测试集上的指标展示
def test_metric_show(var1,var2,var3,var4):
    with st.container():
        st.write("三、在测试集上的指标展示：")
        with st.expander("ℹ️ 指标说明"):
            st.info('Mean Rank：正确实体的平均排名，该值越小越好;\n\n\n\n\n\n\nhit@10：正确实体排名在前10的比例，该值越大越好；')
        st.write(pd.DataFrame({
            "mean_rank(raw)": [var1],
            "hit@10(raw)": [var2],
            "mean_rank(filtered)": [var3],
            "hit@10(filtered)": [var4]
        }))

# 压缩包下载
def zip_download_button(zip_file):
    with st.container():
        st.write("四、训练完成的结果下载")
        with open(zip_file, "rb") as f:
            return st.download_button(
                label="✨ 结果下载",
                data=f,
                file_name="all_results.zip",
                mime="application/zip"
            )

st.set_page_config(
   page_title="Trans-Series Knowledge Graph Embedding",
   page_icon="🎈",
   layout="wide",
   initial_sidebar_state="expanded",
)

# set app layout width
def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
_max_width_()

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    image = Image.open(r'D:\Codes\网安综合实验\Main\logo.png')
    st.image(image, width=400)
    st.header("")

with c32:
    st.header("")
    st.header("")
    st.header("")
    st.text("")
    st.text("")
    st.header("")

st.markdown("## **Introduction**")

with st.expander("ℹ️ - About this app"):
    st.write(
        """     
1.  基于Translation的知识图谱表示学习经典模型，包含TransE、TransH、TransR、TransD;
2.  左侧参数设置栏进行模型选择、数据集选择和训练参数设置;
3.  右侧区域会显示如下相关信息:
-   所选算法的主要思想
-   所选算法的核心代码
-   所选数据集的基本情况
-   训练过程的损失函数可视化
-   在测试集上的指标展示
-   结果下载
	    """
)

    st.markdown("")

with st.expander("🔆 Coming soon!", expanded=False):
    st.write(
        """  
-   Add more knowledge graph embedding models.
-   Add more available dataset.
-   Allow uploading user's personal dataset.

	    """
    )
    st.markdown("")

st.markdown("")

# 左侧侧边栏
with st.sidebar:
    st.markdown("# ℹ️参数设置")
    # st.header("知识图谱Trans系列表示学习算法")
    with st.form("params_form"):
        # radio单选框会返回所选择的选项内容
        st.write("1.算法选择")
        algorithm_selection = st.radio(
            "选择要使用的算法：",
            ('TransE',"TransH","TransR","TransD"),
            help="点击底部运行按钮后，会在右侧区域展示所选算法的相关介绍"
        )
        st.write("2.数据集选择")
        dataset_selection = st.radio(
            "选择默认提供的数据集：",
            ('FreeBase15K', "WordNet18", "Countries","DBpedia50","WN18RR","UMLS"),
            help="点击底部运行按钮后，会在右侧区域展示所选数据集的相关介绍"
        )

        # number_input框会返回输入的数字
        st.write("3.Embedding维度设置")
        embedding_selection = st.number_input(
            label="请设置embedding维度"
        )

        st.write("4.学习率设置")
        lr_selection = st.number_input(
            label="请设置算法学习率",
            min_value=0.01
        )

        st.write("5.epoches设置")
        epoch_selection = st.number_input(
            label="请设置epoch的数量"
        )

        st.write("6.训练完成后是否在测试集上进行测试")
        test_flag = st.selectbox(
            label=" ",
            options =(True,False),
            help="NOTE: 会花费较长时间!"
        )

        # 用户选择的参数列表
        # 同时对输入的变量类型进行必要转换
        para_list = [
            algorithm_selection,
            dataset_selection,
            int(embedding_selection),
            lr_selection,
            int(epoch_selection),
            test_flag
        ]
        st.info("点击开始运行按钮前请再确认一下所选参数噢~")
        submitted = st.form_submit_button("开始运行",help="请注意确认参数")


# 右侧主内容区域
st.markdown("## **Relative Information**")

# 点击submit按钮之后的操作
if submitted:
    # 获取各个参数值，方便后续传入模型
    alg = para_list[0]
    dataset = para_list[1]
    embedding_size = para_list[2]
    lr_rate = para_list[3]
    epoch_nums = para_list[4]
    testFlage = para_list[-1]

    with st.expander("1.参数设置情况"):
        st.info(f"（1）使用算法：{alg}\n\n（2）选用数据集：{dataset}\n\n（3）嵌入维度：{embedding_size}\n\n（4）设置的学习率：{lr_rate}\n\n（5）设置的迭代轮数：{epoch_nums}")

    alg_introduction(alg)
    dataset_introduction(dataset)
    alg_sourcecode_show(alg)

# 如果是TransE算法
    if algorithm_selection == "TransE":
        data_paths = ds_selection(dataset)
        batch_size = batchSize_selection(dataset)
        trainTriples, entityToId, relationToId, validTriples, testTriples = data_paths["train"], data_paths["entity2id"], data_paths["relation2id"], data_paths["valid"], data_paths["test"]
        entity_set, relation_set, triple_list, valid_triple_list = TransE_pytorch.dataloader(trainTriples, entityToId, relationToId, validTriples)
        csv_filePath = "param.csv"
        fig_path = f"{dataset}_torch_TransE_loss_plot.png"
        entityEmbRes = f"{dataset}_torch_TransE_entity_{embedding_size}dim_batch{batch_size}"
        relationEmbRes = f"{dataset}_torch_TransE_relation_{embedding_size}dim_batch{batch_size}"
        transE = TransE_pytorch.TransE(entity_set, relation_set, triple_list, embedding_dim=embedding_size, lr=lr_rate, margin=6.0,
                        norm=1, C=0.25,
                        valid_triple_list=valid_triple_list)
        transE.data_initialise()
        st.markdown("## **Check & Download results**")
        st.success('-------------------------Start Training------------------------')
        df = transE.training_run(epochs=epoch_nums, batch_size=batch_size, out_file_title=f"{dataset}_torch_")
        df.to_csv("param.csv")
        if testFlage == True:
            st.success('-------------------------Start Testing------------------------')
            transE.insert_test_data(entityEmbRes, relationEmbRes, testTriples)
            raw_hit_10, raw_mean_rank = transE.test_run(filter=False)
            filtered_hit_10, filtered_mean_rank = transE.test_run(filter=True)

        st.success('-------------------------😊😊Belows are results😊😊------------------------')
        params_show(csv_filePath)
        loss_plot(fig_path)
        if raw_hit_10 and raw_mean_rank and filtered_mean_rank and filtered_hit_10:
            test_metric_show(raw_mean_rank,raw_hit_10,filtered_mean_rank,filtered_hit_10)
        zipObj = ZipFile("results.zip","w")
        zipObj.write(entityEmbRes)
        zipObj.write(relationEmbRes)
        zipObj.close()
        ZipfileDotZip = "results.zip"
        zip_download_button(ZipfileDotZip)

        st.success(f"Please use above button to download embedding result~")
        st.success("---------------------------END----------------------------")

# 如果是TransH算法：
    if algorithm_selection == "TransH":
        data_paths = ds_selection(dataset)
        batch_size = batchSize_selection(dataset)
        embedding_dim = embedding_size
        trainTriples, entityToId, relationToId, validTriples, testTriples = data_paths["train"], data_paths[
                "entity2id"], data_paths["relation2id"], data_paths["valid"], data_paths["test"]
        entity_set, relation_set, triple_list, valid_triple_list = TransH_pytorch.dataloader(trainTriples,
                                                                                                 entityToId,
                                                                                                 relationToId,
                                                                                                 validTriples)
        csv_filePath = "param.csv"
        fig_path = f"{dataset}_torch_TransH_loss_plot.png"
        entityEmbRes = f"{dataset}_torch_TransH_entity_{embedding_dim}dim_batch{batch_size}"
        normRelations = f"{dataset}_torch_TransH_norm_relations_{embedding_dim}dim_batch{batch_size}"
        hyperRelations = f"{dataset}_torch_TransH_hyper_relations_{embedding_dim}dim_batch{batch_size}"
        lossRecord = f"{dataset}_torch_loss_record.txt"
        transH = TransH_pytorch.TransH(entity_set, relation_set, triple_list, embedding_dim=embedding_dim,
                                           lr=lr_rate, margin=8.0, norm=1, C=1.0, epsilon=1e-5,
                                           valid_triple_list=valid_triple_list)
        transH.data_initialise()
        st.markdown("## **Check & Download results**")
        st.success('-------------------------Start Training------------------------')
        df = transH.training_run(epochs=epoch_nums, batch_size=batch_size, out_file_title=f"{dataset}_torch_")
        df.to_csv("param.csv")
        if testFlage == True:
            st.success('-------------------------Start Testing------------------------')
            transH.insert_data(entityEmbRes, normRelations, hyperRelations, testTriples, lossRecord)
            raw_hit_10, raw_mean_rank = transH.test_run(filter=False)
            filtered_hit_10, filtered_mean_rank = transH.test_run(filter=True)

        st.success('-------------------------😊😊Belows are results😊😊------------------------')
        params_show(csv_filePath)
        loss_plot(fig_path)
        if raw_hit_10 and raw_mean_rank and filtered_mean_rank and filtered_hit_10:
            test_metric_show(raw_mean_rank, raw_hit_10, filtered_mean_rank, filtered_hit_10)
        zipObj = ZipFile("results.zip", "w")
        zipObj.write(entityEmbRes)
        zipObj.write(normRelations)
        zipObj.write(hyperRelations)
        zipObj.close()
        ZipfileDotZip = "results.zip"
        zip_download_button(ZipfileDotZip)

        st.success(f"Please use above button to download embedding result~")
        st.success("---------------------------END----------------------------")

# 如果是TransR算法
    if algorithm_selection == "TransR":
        data_paths = ds_selection(dataset)
        batch_size = batchSize_selection(dataset)
        entity_embedding_dim = embedding_size
        relation_embedding_dim = embedding_size
        trainTriples, entityToId, relationToId, validTriples, testTriples = data_paths["train"], data_paths["entity2id"], data_paths["relation2id"], data_paths["valid"], data_paths["test"]
        entity_set, relation_set, triple_list, valid_triple_list = TransR_pytorch.dataloader(trainTriples, entityToId, relationToId, validTriples)
        csv_filePath = "param.csv"
        fig_path = f"{dataset}_torch_TransR_loss_plot.png"
        entityEmbRes = f"{dataset}_torch_TransR_entity_{embedding_size}dim_batch{batch_size}"
        relationEmbRes = f"{dataset}_torch_TransR_relation_{embedding_size}dim_batch{batch_size}"
        rel_matrix = f"{dataset}_torch_TransR_rel_matrix_{entity_embedding_dim}_{relation_embedding_dim}dim_batch{batch_size}"
        transR = TransR_pytorch.TransR(entity_set, relation_set, triple_list, ent_dim=entity_embedding_dim, rel_dim=relation_embedding_dim, lr=lr_rate, margin=6.0,
                                 norm=1, C=0.25, valid_triples=valid_triple_list)
        transR.data_initialise()
        st.markdown("## **Check & Download results**")
        st.success('-------------------------Start Training------------------------')
        df = transR.training_run(epochs=epoch_nums, batch_size=batch_size, out_file_title=f"{dataset}_torch_")
        df.to_csv("param.csv")
        if testFlage == True:
            st.success('-------------------------Start Testing------------------------')
            transR.insert_test_data(entityEmbRes, relationEmbRes, rel_matrix,testTriples)
            raw_hit_10, raw_mean_rank = transR.test_run(filter=False)
            filtered_hit_10, filtered_mean_rank = transR.test_run(filter=True)

        st.success('-------------------------😊😊Belows are results😊😊------------------------')
        params_show(csv_filePath)
        loss_plot(fig_path)
        if raw_hit_10 and raw_mean_rank and filtered_mean_rank and filtered_hit_10:
            test_metric_show(raw_mean_rank,raw_hit_10,filtered_mean_rank,filtered_hit_10)
        zipObj = ZipFile("results.zip", "w")
        zipObj.write(entityEmbRes)
        zipObj.write(relationEmbRes)
        zipObj.write(rel_matrix)
        zipObj.close()
        ZipfileDotZip = "results.zip"
        zip_download_button(ZipfileDotZip)

        st.success(f"Please use above button to download embedding result~")
        st.success("---------------------------END----------------------------")

# 如果是TransD算法：
    if algorithm_selection == "TransD":
        data_paths = ds_selection(dataset)
        batch_size = batchSize_selection(dataset)
        entity_embedding_dim = embedding_size
        relation_embedding_dim = embedding_size
        trainTriples, entityToId, relationToId, validTriples, testTriples = data_paths["train"], data_paths["entity2id"], data_paths["relation2id"], data_paths["valid"], data_paths["test"]
        entity_set, relation_set, triple_list, valid_triple_list = TransD_pytorch.dataloader(trainTriples, entityToId, relationToId, validTriples)
        csv_filePath = "param.csv"
        fig_path = f"{dataset}_torch_TransD_loss_plot.png"
        entityEmbRes = f"{dataset}_torch_TransD_entity_{embedding_size}dim_batch{batch_size}"
        relationEmbRes = f"{dataset}_torch_TransD_relation_{embedding_size}dim_batch{batch_size}"
        entity_transfer = f"{dataset}_torch_TransD_ent_transfer_{entity_embedding_dim}_{relation_embedding_dim}dim_batch{batch_size}"
        relation_transfer = f"{dataset}_torch_TransD_rel_transfer_{entity_embedding_dim}_{relation_embedding_dim}dim_batch{batch_size}"
        transD = TransD_pytorch.TransD(entity_set, relation_set, triple_list, embedding_dim=entity_embedding_dim, lr=lr_rate, margin=4.0, norm=1, C = 0.25, valid_triple_list=valid_triple_list)
        transD.data_initialise()
        st.markdown("## **Check & Download results**")
        st.success('-------------------------Start Training------------------------')
        df = transD.training_run(epochs=epoch_nums, batch_size=batch_size, out_file_title=f"{dataset}_torch_")
        df.to_csv("param.csv")
        if testFlage == True:
            st.success('-------------------------Start Testing------------------------')
            transD.insert_test_data(entityEmbRes, relationEmbRes, relation_transfer, entity_transfer,testTriples)
            raw_hit_10, raw_mean_rank = transD.test_run(filter=False)
            filtered_hit_10, filtered_mean_rank = transD.test_run(filter=True)

        st.success('-------------------------😊😊Belows are results😊😊------------------------')
        params_show(csv_filePath)
        loss_plot(fig_path)
        if raw_hit_10 and raw_mean_rank and filtered_mean_rank and filtered_hit_10:
            test_metric_show(raw_mean_rank,raw_hit_10,filtered_mean_rank,filtered_hit_10)
        zipObj = ZipFile("results.zip", "w")
        zipObj.write(entityEmbRes)
        zipObj.write(relationEmbRes)
        zipObj.write(entity_transfer)
        zipObj.write(relation_transfer)
        zipObj.close()
        ZipfileDotZip = "results.zip"
        zip_download_button(ZipfileDotZip)

        st.success(f"Please use above button to download embedding result~")
        st.success("---------------------------END----------------------------")

























