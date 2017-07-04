# coding=gbk
"""
    �����ݼ����г�����ͳ��
"""
import stanford_parser as sp
import pandas as pd
import utils.data_path as path


def read_dataset_split(filename):
    """
    ��ȡdatasetSplit�ļ���ͳ��ѵ�����������Լ���֤������
    :param filename:�ļ��� 
    :return: result��list�±����datasetSentences�ж�Ӧ���ݵı�ţ�ע��list��0��ʼ��datasetSentences��1��ʼ
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
    ��ȡ�����ļ���ÿ������һ������
    :param filename: ����·��
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
    ��ȡ�ֵ��ļ�����ע��������ӡ����ӽ�����Stanford parse��
    :param filename: �ֵ�·��
    :param sentiment_dict:��read_sentiment_labels�ж�ȡ�����ݣ�key��id��values�Ƕ�Ӧ����������������ʾ
    :return: dict�����ֵ����ʽ�洢��key�����ݣ�values�Ƕ�Ӧ�ķ������
    """
    diction = dict()
    lines = open(filename, encoding="utf8").readlines()
    for i in lines:
        u = i.replace("\n", "").split("|")
        diction[u[0]] = sentiment_dict[u[1]]
    return diction


def read_sentiment_labels(filename):
    """
    ��ȡ��б�ǩ���ݣ�������б�ǩid��Ӧ��dictionary�е�id������score�ж�����һ�����
    :return: dict��key��id��values�Ƕ�Ӧ����������������ʾ
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

    #��ȡ�����������ļ���Ӧ����pandas dataframe����
    dataset_sentences = read_dataset_sentences(path.DATASET_SENTENCES)
    pd_data['sentence'] = dataset_sentences

    #��ȡ������Ӧ����б�ǩ
    sentiment_labels = read_sentiment_labels(path.SENTIMENT_LABELS)
    #���ֵ�תΪ��Ӧ����б�ǩ����
    dictionary = read_dictionary(path.DICTIONARY,sentiment_labels)

    #��ȡ���ӵ���б�ǩ
    print(pd_data)

    def f(x):
        if x in dictionary:
            return dictionary[x]
        else:
            return 0
    pd_data['sentiment_label'] = pd_data['sentence'].apply(f)

    print(pd_data)

    #ȥ����б�ǩΪNone����
    temp = pd_data[pd_data['sentiment_label'] != 0]
    print(temp)

    return temp


def parse_sentence(pd_data,target_file):
    """
    �����ݽ��н���
    :param pd_data:pandas dataframe
    :return:���������������
    """
    pd_data['parse'] = pd_data['sentence'].apply(sp.getAllConstituents)
    print(pd_data)

    # д���ļ���
    if target_file:
        pd_data.to_csv(target_file, encoding='utf8')


if __name__ == '__main__':
    parse_sentence(get_train_test_data(),path.TARGET_CSV)


"""
train has 8544 num, test has 2210 num, dev has 1101 num
"""

"""
target.csv�а�����ѵ���������Լ�����֤��
"""