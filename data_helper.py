# coding=utf8
"""
    数据获取类，包含了加载原始数据数据（为csv格式），生成embedding等
"""
import re
import numpy as np
import pandas as pd

def clean_str(s):
    # 将符合下面正则表达式的部分替换为空格、:、\'s等
    s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)  #将除了非数字、非大小写字母、非:(),!?'`的部分都替换为空格
    s = re.sub(r" : ", ":", s)   #将:前后的空格替换为:
    s = re.sub(r"\'s", " \'s", s)   #将类似于it's替换为空格+'s
    s = re.sub(r"\'ve", " \'ve", s)   #i've替换为空格+'ve
    s = re.sub(r"n\'t", " n\'t", s)  #
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)   #将,分隔开来
    s = re.sub(r"!", " ! ", s)   #将!分隔开
    s = re.sub(r"\(", " \( ", s)  #将(分隔开
    s = re.sub(r"\)", " \) ", s)  #将)分隔开
    s = re.sub(r"\?", " \? ", s)  #将?分隔开
    s = re.sub(r"\s{2,}", " ", s)  #{2,}表示精确匹配n个前面表达式，也就是将多个空格、制表、换页替换为空格
    return s.strip().lower()  #默认字符串前后删除空白符（包括'\n', '\r',  '\t',  ' ')，并转为小写


def load_data(filename, compression = None):
    """
    需要加载的文件名
    :param filename:zip文件
    :return:
    返回的6个值分别为：
    x:文本向量，实际上是序号向量
    y:标签向量，one-hot向量，长度为总标签个数
    vocabulary:单词以及对应的序号，词典
    vocabulary_inv:所有单词
    df:原始文本和标签，剔除不相关的列，并进行了乱序，注意这里仍然是原始文本，但是行与x、y是一一对应的
    labels:对应的标签
    """
    if not compression:
        df = pd.read_csv(filename,index_col=0,encoding="utf8")
    elif compression and compression in ['zip']:
        df = pd.read_csv(filename,index_col=0,encoding="utf8",compression=compression)
    else:
        raise Exception("Attention, compression should be recognized by pandas!")

    selected = ['dataset_label','sentence','sentence_parse','category']   #原始数据的几个标签

    non_selected = list(set(df.columns) - set(selected))   # 集合运算

    df = df.drop(non_selected, axis=1)  # 删掉指定列的每一行
    df = df.dropna(axis=0, how='any', subset=selected)  # 删去没有值，即为None的列
    """
    permutation为序列的意思
    参数x : int or array_like
    If `x` is an integer, randomly permute ``np.arange(x)``.
    If `x` is an array, make a copy and shuffle the elements randomly.
    """
    df = df.reindex(np.random.permutation(df.index))   #相当于乱序，也可以用sklearn的shuffle

    labels = sorted(list(set(df[selected[-1]].tolist())))   #去掉重复标签
    num_labels = len(labels)  #总共的标签数量
    one_hot = np.zeros((num_labels, num_labels), int)   #产生int类型的0矩阵
    np.fill_diagonal(one_hot, 1)   #填充为对角矩阵，无返回值

    """
    下面这句代码似乎有问题
    如果zip两个列表，可以将zip后的转为dict，但如果是一个列表一个矩阵，似乎不能转为dict
    原意应该是为了将标签转为对应的标签向量
    """
    label_dict = dict(zip(labels, one_hot))

    """
    清理每份文本，并用空格分隔后转为list
    这个部分也可以使用斯坦福NLP进行分词
    """
    x_raw= df[selected[1]].apply(lambda x: clean_str(x).split(' ')).tolist()

    # 取出df[selected[0]]即每行对应的标签，找到对应的标签向量
    y_raw = df[selected[-1]].apply(lambda y: label_dict[y]).tolist()

    #填充sentence
    x_raw = pad_sentences(x_raw)

    #vocabulary是dict，词与对应的序号；vocabulary_inv表示词。注意：这里是经过填充后再进行建立词典，因此填充的词仍然在这里面
    vocabulary, vocabulary_inv = build_vocab(x_raw)

    #这里建立每一个文本对应的序号向量以及标签向量，注意：这里都是np.array向量
    x = np.array([[vocabulary[word] for word in sentence] for sentence in x_raw])
    y = np.array(y_raw)

    return x, y, vocabulary, vocabulary_inv, df, labels