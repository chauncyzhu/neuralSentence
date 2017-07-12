# coding=utf8
"""
    rnn的类
"""
from tensorflow.contrib import rnn
import tensorflow as tf
class TextRNN(object):
    def __init__(self,embedding_mat,non_static,hidden_unit,sequence_length,num_classes,
                 embedding_size,l2_reg_lambda):
        """
        rnn需要的一些参数，根据这设置一些计算节点
        :param embedding_mat:embedding矩阵
        :param non_static:是否静态，否（non_static=True）意味着需要更新embedding，是（non_static=False）意为不需要更新embedding
        :param hidden_unit:隐藏层单元的个数
        :param sequence_length:序列长度
        :param num_classes:类别的个数
        :param embedding_size:embedding维度大小
        :param l2_reg_lambda:l2正则
        """
        # 第一个参数说明输入的类型，这里的shape不需要embedding_size，以为后面会用到查找表
        self.input_x = tf.placeholder(tf.int32,[None,sequence_length],name="input_x")
        self.input_y = tf.placeholder(tf.int32,[None,num_classes],name="input_y")
        # dropout_keep_prob概率大小
        self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")
        # 每一个batch_size的大小
        self.batch_size = tf.placeholder(tf.int32, [])

        l2_loss = tf.constant(0.0)

        # 创建embedding计算节点
        with tf.device("/cpu0:"),tf.name_scope("embedding"):
            if not non_static:  #如果non_static为false，也就是为静态的static的
                W = tf.constant(embedding_mat, name="W")
            else:
                W = tf.Variable(embedding_mat, name="W")

            self.embedding_chars = tf.nn.embedding_lookup(W,self.input_x)

            emb = tf.expand_dims(self.embedding_chars,-1)

        # 创建LSTM节点
        lstm_cell = rnn.BasicLSTMCell(num_units=hidden_unit,forget_bias=1.0)
        lstm_cell = rnn.DropoutWrapper(lstm_cell,output_keep_prob=self.dropout_keep_prob)

        # 为什么这个batch_size需要以这种方式传入？
        self._initial_state = lstm_cell.zero_state(self.batch_size,tf.float32)

        # 输出，这里的initial_state表明lstm结构中的初始状态，这里的sequence_length是否可以不给出？
        outputs, states = rnn.static_rnn(lstm_cell,self.input_x,initial_state=self._initial_state,dtype=tf.float32)

        """
        暂时不采用更复杂的实现
        """
        output = outputs[-1]   # 选择最后一个节点作为输出

        with tf.name_scope("output"):
            # 输出层的W
            self.W = tf.Variable(tf.truncated_normal([hidden_unit,num_classes],stddev=0.1),name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_classes],name='b'))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(output,self.W,b, name='scores')   # x*weights+b
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
            self.loss = tf.reduce_mean(losses)  + l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')

        # 正确的个数
        with tf.name_scope('num_correct'):
            correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct, 'float'))











