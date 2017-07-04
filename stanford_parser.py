# coding=gbk
"""
    ʹ��stanford parser���н���
"""
import os
import numpy as np
import utils.data_path as path
import nltk.tree
from nltk import word_tokenize
from nltk.parse import stanford

#���stanford��������,�˴���Ҫ�ֶ��޸ģ�jar����ַΪ���Ե�ַ��
os.environ['STANFORD_PARSER'] = path.STANFORD_PARSER
os.environ['STANFORD_MODELS'] = path.STANFORD_MODELS
#ΪJAVAHOME��ӻ�������
java_path = path.JAVA_PATH
os.environ['JAVAHOME'] = java_path
PAESER = stanford.StanfordParser(model_path=path.ENGLISHPCFG)

#���height>2����ݹ���÷���
def __getLeaves(node,constituents):
    """
    ��ȡ���е�constituents�����ڵ��Ӧ�����д��Լ�������Ӧ�����д�
    :param node: ����parse��tree
    :param constituents: ����constituency parse
    :return: constituents--ע�ⲻ����������
    """
    if node.height() == 2:
        # print("node:", node)
        # print("node label:", node.label())
        constituents.append([node.leaves()[0],node.label()])
    elif node.height() > 2 and (node.label() == 'NP' or 'VP'):
        for sub_node in node:
            __getLeaves(sub_node, constituents)
    else:
        return constituents

def getAllConstituents(single_line):
    """
    ��single line���н���
    :param parser: ����parser
    :param single_line: string
    :return:
    """
    # �䷨��ע
    parse_line = PAESER.raw_parse(single_line)
    constituents = []
    print("constituency parse by stanford corenlp")
    for line in parse_line:
        #line.draw()
        for root in line:
            #print("parse tree:", root)
            for tree in root:
                __getLeaves(tree, constituents)
    return constituents

if __name__ == '__main__':
    single_line = "during the mass high school education movement from 1910-1940, there was an increase in skilled workers"

    constituents_list = getAllConstituents(single_line)  #ע�⣬����ʵ����������Ƕ���б���Ҫ����ѭ�����н���
    print(constituents_list)
