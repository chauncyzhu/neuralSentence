# coding=gbk
"""
    对数据集进行初步的统计
"""
import stanford_parser as sp
import pandas as pd
import utils.data_path as path


def read_dataset_split(filename):
    """
    读取datasetSplit文件，统计训练集、测试以及验证集数量
    :param filename:文件名 
    :return: result，list下标就是datasetSentences中对应数据的编号，注意list从0开始，datasetSentences从1开始
    """
    lines = open(filename,encoding="utf8").readlines()
    result = []
    for line in lines[1:]:
        line = line.strip()
        line = line.split(",")
        print(line)
        result.append(int(line[1]))

    count = [0] * 3
    for i in result:
        if i == 1:
            count[0] += 1
        if i == 2:
            count[1] += 1
        if i == 3:
            count[2] += 1
    print("train has %d num, test has %d num, dev has %d num" % (count[0], count[1], count[2]))
    return result

def read_dataset_sentences(filename):
    """
    读取数据文件，每个都是一个句子
    :param filename: 数据路径
    :return: 
    """
    lines = open(filename,encoding="utf8").readlines()
    result = []
    for line in lines[1:]:
        line = line.strip()
        line = line.split("\t")
        print(line)
        result.append(line[1])

    return result

def read_dictionary(filename,sentiment_dict):
    """
    读取字典文件，标注了整句句子、句子解析（Stanford parse）
    :param filename: 字典路径
    :param sentiment_dict:从read_sentiment_labels中读取的数据，key是id，values是对应的情感类别，用向量表示
    :return: dict，用字典的形式存储，key是内容，values是对应的分类类别
    """
    diction = dict()
    lines = open(filename, encoding="utf8").readlines()
    for i in lines:
        u = i.replace("\n", "").split("|")
        diction[u[0]] = sentiment_dict[u[1]]
    return diction


def read_sentiment_labels(filename):
    """
    读取情感标签数据，其中情感标签id对应了dictionary中的id，根据score判断是哪一类情感
    :return: dict，key是id，values是对应的情感类别，用向量表示
    """
    diction = dict()
    lines = open(filename, encoding="utf8").readlines()
    for i in lines[1:]:
        u = i.replace("\n", "").split("|")
        nu = float(u[1])
        if nu <= 0.2:
            uu = [1,0,0,0,0]
        elif nu <= 0.4:
            uu = [0,1,0,0,0]
        elif nu <= 0.6:
            uu = [0,0,1,0,0]
        elif nu <= 0.8:
            uu = [0,0,0,1,0]
        else:
            uu = [0,0,0,0,1]
        diction[u[0]] = uu

    return diction


def get_train_test_data():
    pd_data = pd.DataFrame(columns=['id','dataset_label','sentence','sentiment_label'])

    dataset_split = read_dataset_split(path.DATASET_SPLIT)
    pd_data['id'] = range(1,len(dataset_split)+1)
    pd_data['dataset_label'] = dataset_split

    #读取完整的数据文件，应该是pandas dataframe类型
    dataset_sentences = read_dataset_sentences(path.DATASET_SENTENCES)
    pd_data['sentence'] = dataset_sentences

    #获取句子相应的情感标签
    sentiment_labels = read_sentiment_labels(path.SENTIMENT_LABELS)
    #将字典转为对应的情感标签向量
    dictionary = read_dictionary(path.DICTIONARY,sentiment_labels)

    #获取句子的情感标签
    print(pd_data)

    def f(x):
        if x in dictionary:
            return dictionary[x]
        else:
            return 0
    pd_data['sentiment_label'] = pd_data['sentence'].apply(f)

    print(pd_data)

    #去掉情感标签为None的行
    temp = pd_data[pd_data['sentiment_label'] != 0]
    print(temp)

    return temp


def parse_sentence(pd_data,target_file):
    """
    对数据进行解析
    :param pd_data:pandas dataframe
    :return:经过解析后的数据
    """
    pd_data['parse'] = pd_data['sentence'].apply(sp.getAllConstituents)
    print(pd_data)

    # 写入文件中
    if target_file:
        pd_data.to_csv(target_file, encoding='utf8')


if __name__ == '__main__':
    parse_sentence(get_train_test_data(),path.TARGET_CSV)


"""
train has 8544 num, test has 2210 num, dev has 1101 num
"""

"""
target.csv中包含了训练集、测试集、验证集
"""