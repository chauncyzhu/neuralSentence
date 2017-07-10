# coding=gbk
"""
    文件目录路径
"""
ROOT_PATH = "D:/Data/sentiment_treebank/"

DATASET_SENTENCES = ROOT_PATH + "datasetSentences.txt"
DICTIONARY = ROOT_PATH + "dictionary.txt"
DATASET_SPLIT = ROOT_PATH + "datasetSplit.txt"
SENTIMENT_LABELS = ROOT_PATH + "sentiment_labels.txt"

TARGET_CSV = ROOT_PATH + "target.csv"

# stanford nlp配置
STANFORD_PATH = "D:/Coding/pycharm-professional/nltk/stanford_nlp/parser/"
JAVA_PATH = "C:/Program Files/Java/jdk1.8.0_121/bin/java.exe"

STANFORD_PARSER = STANFORD_PATH + "stanford-parser.jar"
STANFORD_MODELS = STANFORD_PATH + "stanford-parser-3.7.0-models.jar"
ENGLISHPCFG = STANFORD_PATH + "englishPCFG.ser.gz"

# 预训练的word2vec词向量
WORD2VEC_BIN = "D:/Data/pretrained_vec/GNv_NS300d.bin"

#词典文件
WORD2VEC_DICT = ROOT_PATH + "word2vec_dict.csv"

