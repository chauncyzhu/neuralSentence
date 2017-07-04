# coding=gbk
"""
    使用stanford parser进行解析
"""
import os
import numpy as np
import utils.data_path as path
import nltk.tree
from nltk import word_tokenize
from nltk.parse import stanford

#添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
os.environ['STANFORD_PARSER'] = path.STANFORD_PARSER
os.environ['STANFORD_MODELS'] = path.STANFORD_MODELS
#为JAVAHOME添加环境变量
java_path = path.JAVA_PATH
os.environ['JAVAHOME'] = java_path
PAESER = stanford.StanfordParser(model_path=path.ENGLISHPCFG)

#如果height>2，则递归调用方法
def __getLeaves(node,constituents):
    """
    获取所有的constituents，根节点对应的所有词以及子树对应的所有词
    :param node: 经过parse的tree
    :param constituents: 属于constituency parse
    :return: constituents--注意不包括单个词
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
    对single line进行解析
    :param parser: 传入parser
    :param single_line: string
    :return:
    """
    # 句法标注
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

    constituents_list = getAllConstituents(single_line)  #注意，这里实际上是三层嵌套列表，需要两个循环进行解析
    print(constituents_list)
