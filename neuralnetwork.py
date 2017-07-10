# coding=gbk
"""
    �������������Խṹ������Ϣ����
"""
import numpy as np
import pandas as pd
import preprocess as pre
import utils.data_path as path
import tensorflow as tf
from tensorflow.contrib import rnn

#��������
learning_rate = 0.001
training_iters = 100
batch_size = 119 #�������е�����
display_step = 10

#���������
n_input = 1  #�������ݵ�ά��
n_steps = 187  #RNN��ʱ����չ����ά��
n_hidden = 187  #���������
n_classes = 2  #��������Ҳ������ά��

#�����ֵ
(x_train,y_train,x_test,y_test) = id.read_csv()  #ÿһ��������187ά�����������

#���ü���ͼ
x = tf.placeholder("float32",[None,n_steps,n_input])
y = tf.placeholder("float32",[None,n_classes])

#Ȩֵ��ƫ����
weights = {"out_weight":tf.Variable(tf.random_normal([n_hidden,n_classes]))}
biases = {"out_bias":tf.Variable(tf.random_normal([n_classes]))}

def RNN(x,weights,biases):
    """
    RNN�ṹ
    :param x:�����xֵ
    :param weights:Ȩ��
    :param biases:ƫ��
    :return:
    """
    x = tf.unstack(x, n_steps, 1)
    lstm_cell = rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)   #LSTM��Ԫ
    outputs,states = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)  #���LSTM�����
    #����outputs[-1]ȡ�������һ�������
    return tf.matmul(outputs[-1],weights['out_weight']) + biases['out_bias']   #rnn�����

train = RNN(x,weights,biases)   #����·��ѵ��

#������ʧ�������Ż���ʽ
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train,labels=y))  #������
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#����ģ��
correct_train = tf.equal(tf.arg_max(train,1),tf.arg_max(y,1))  #���ֵ���ڵ�����Ƿ���ȣ������1��dimension�����廹��֪��ʲô��˼
accuracy = tf.reduce_mean(tf.cast(correct_train,tf.float32))  #ת��float32��Ϊ�˸���������������reduce_mean��Ϊ��ȡ��ֵ

#������ʼ��
init = tf.global_variables_initializer()

#��session������
with tf.Session() as sess:
    sess.run(init)  #���г�ʼ������
    #��ʼѵ��
    for i in range(training_iters):
        x_train = x_train.reshape((batch_size,n_steps,n_input))
        sess.run(train,feed_dict={x:x_train,y:y_train})
        if i % display_step == 0:
            #��ʾ��ǰ��׼ȷ��
            acc = sess.run(accuracy,feed_dict={x:x_train,y:y_train})
            #��ǰ��ʧ����
            loss = sess.run(cost, feed_dict={x:x_train,y:y_train})
            print("Iter " + str(i) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
    print("Optimization Finished!")

    #��ʼ����
    x_test = x_test.reshape((-1, n_steps, n_input))
    print("Testing Accuracy:",sess.run(accuracy, feed_dict={x: x_test, y: y_test}))
    print("Testing Accuracy:",sess.run(train, feed_dict={x: x_test, y: y_test}))


if __name__ == '__main__':
    pass


"""
2621 words in original data but not in word2vec vocabulary
pd dict has 20866 words in total
"""