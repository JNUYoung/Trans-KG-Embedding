import streamlit as st
import pandas as pd
import os.path as osp
import sys

# 一、参数变化情况展示函数
def params_show(file):
    st.write("一、算法训练过程中的参数展示：")
    _df = pd.read_csv(csv_filePath, index_col=0)
    st.dataframe(data=_df, width=1000, height=200)

# 二、损失函数变化情况展示函数
def loss_plot(file):
    with st.container():
        st.write("二、测试集和训练集的损失函数变化情况：")
        # st.line_chart(chart_data)
        # 通过绝对路径读取训练的loss变化曲线
        image = Image.open(fig_path)
        st.image(image, caption="损失函数变化情况")

# 三、测试集上的指标展示
def test_metric_show(var1,var2,var3,var4):
    with st.container():
        st.write("三、在测试集上的指标展示：")
        st.info('Mean Rank：正确实体的平均排名，该值越小越好;\n\n\n\n\n\n\nhit@10：正确实体排名在前10的比例，该值越大越好；')
        st.write(pd.DataFrame({
            "mean_rank(raw)": [var1],
            "hit@10(raw)": [var2],
            "mean_rank(filtered)": [var3],
            "hit@10(filtered)": [var4]
        }))

# 训练完成后的结果下载
# def convert_df(df):
#     return df.to_csv().encode('utf-8')
#
# def file_save(file_path):
#     with codecs.open(file_path,"r") as f1:
#         content = f1.readlines()
#         with open("result.txt", "w") as f2:
#             for line in content:
#                 f2.write(f"{line}\n")
#     return "result.txt"

def click_fun(aaa):
    st.success(f"Successfully download {aaa} embedding result😀~")


def result_download(file_path,aaaa):
    with st.container():
        if aaaa == "entity":
            st.write("四、训练完成后的实体embedding结果下载：")
            with open(file_path, "r") as file:
                return st.download_button(
                    label="下载实体的embedding结果",
                    # data=ent_res_path,
                    data=file,
                    file_name=f"{aaaa}-download.txt",
                    # on_click=click_fun("entity")
                )
        if aaaa == "relation":
            st.write("五、训练完成后的关系embedding结果下载：")
            with open(file_path, "r") as file:
                return st.download_button(
                    label="下载关系的embedding结果",
                    # data=ent_res_path,
                    data=file,
                    file_name=f"{aaaa}-download.txt",
                    # on_click=click_fun("entity")
                )
        if aaaa == "relation_matrix":
            st.write("六、TransR训练完成后的关系空间映射矩阵：")
            with open(file_path, "r") as file:
                return st.download_button(
                    label="下载TransR模型得到的关系空间映射矩阵",
                    # data=ent_res_path,
                    data=file,
                    file_name=f"{aaaa}-download.txt",
                    # on_click=click_fun("entity")
                )

# 数据集如何选择，以对象的形式返回训练和测试过程所需要的文件的绝对路径
def ds_selection(option):
    if option == "FreeBase15K":
        return {
            "train": r"D:\Codes\网安综合实验\datasets\FB15k\freebase_mtr100_mte100-train.txt",
            "entity2id": r"D:\Codes\网安综合实验\datasets\FB15k\entity2id.txt",
            "relation2id": r"D:\Codes\网安综合实验\datasets\FB15k\relation2id.txt",
            "valid": r"D:\Codes\网安综合实验\datasets\FB15k\freebase_mtr100_mte100-valid.txt",
            "test": r"D:\Codes\网安综合实验\datasets\FB15k\freebase_mtr100_mte100-test.txt"
        }
    if option == "WordNet18":
        return {
            "train": r"D:\Codes\网安综合实验\datasets\WN18\wordnet-mlj12-train.txt",
            "entity2id": r"D:\Codes\网安综合实验\datasets\WN18\entity2id.txt",
            "relation2id": r"D:\Codes\网安综合实验\datasets\WN18\relation2id.txt",
            "valid": r"D:\Codes\网安综合实验\datasets\WN18\wordnet-mlj12-valid.txt",
            "test": r"D:\Codes\网安综合实验\datasets\WN18\wordnet-mlj12-test.txt"
        }
    if option == "Countries":
        return {
            "train": r"D:\Codes\网安综合实验\datasets\Countries\Countries_S1\train.txt",
            "entity2id": r"D:\Codes\网安综合实验\datasets\Countries\Countries_S1\entity2id.txt",
            "relation2id": r"D:\Codes\网安综合实验\datasets\Countries\Countries_S1\relation2id.txt",
            "valid": r"D:\Codes\网安综合实验\datasets\Countries\Countries_S1\valid.txt",
            "test": r"D:\Codes\网安综合实验\datasets\Countries\Countries_S1\test.txt"
        }
    if option == "DBpedia50":
        return {
            "train": r"D:\Codes\网安综合实验\datasets\DBpedia50\train.txt",
            "entity2id": r"D:\Codes\网安综合实验\datasets\DBpedia50\entity2id.txt",
            "relation2id": r"D:\Codes\网安综合实验\datasets\DBpedia50\relation2id.txt",
            "valid": r"D:\Codes\网安综合实验\datasets\DBpedia50\valid.txt",
            "test": r"D:\Codes\网安综合实验\datasets\DBpedia50\test.txt"
        }
    if option == "YAGO3-10":
        return {
            "train": r"D:\Codes\网安综合实验\datasets\YAGO3-10\train.txt",
            "entity2id": r"D:\Codes\网安综合实验\datasets\YAGO3-10\entity2id.txt",
            "relation2id": r"D:\Codes\网安综合实验\datasets\YAGO3-10\relation2id.txt",
            "valid": r"D:\Codes\网安综合实验\datasets\YAGO3-10\valid.txt",
            "test": r"D:\Codes\网安综合实验\datasets\YAGO3-10\test.txt"
        }
    if option == "WN18RR":
        return {
            "train": r"D:\Codes\网安综合实验\datasets\WN18RR\train.txt",
            "entity2id": r"D:\Codes\网安综合实验\datasets\WN18RR\entity2id.txt",
            "relation2id": r"D:\Codes\网安综合实验\datasets\WN18RR\relation2id.txt",
            "valid": r"D:\Codes\网安综合实验\datasets\WN18RR\valid.txt",
            "test": r"D:\Codes\网安综合实验\datasets\WN18RR\test.txt"
        }
    if option == "UMLS":
        return {
            "train": r"D:\Codes\网安综合实验\datasets\UMLS\train.txt",
            "entity2id": r"D:\Codes\网安综合实验\datasets\UMLS\entity2id.txt",
            "relation2id": r"D:\Codes\网安综合实验\datasets\UMLS\relation2id.txt",
            "valid": r"D:\Codes\网安综合实验\datasets\UMLS\valid.txt",
            "test": r"D:\Codes\网安综合实验\datasets\UMLS\test.txt"
        }


def batchSize_selection(input_dataset):
    if input_dataset == "FreeBase15K":
        return 4800
    if input_dataset == "WordNet18":
        return 4800
    if input_dataset == "Countries":
        return 4
    if input_dataset == "DBpedia50":
        return 256
    if input_dataset == "WN18RR":
        return 256
    if input_dataset == "UMLS":
        return 256