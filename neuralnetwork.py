# coding=gbk
"""
    利用神经网络来对结构化的信息分类
"""
import numpy as np
import pandas as pd
import preprocess as pre
import utils.data_path as path
import tensorflow as tf
from tensorflow.contrib import rnn

#参数设置
learning_rate = 0.001
training_iters = 100
batch_size = 119 #就是所有的数据
display_step = 10

#神经网络参数
n_input = 1  #输入数据的维度
n_steps = 187  #RNN在时间上展开的维度
n_hidden = 187  #隐含层个数
n_classes = 2  #类别个数，也是类别的维度

#输入的值
(x_train,y_train,x_test,y_test) = id.read_csv()  #每一行数据有187维，有两种类别

#设置计算图
x = tf.placeholder("float32",[None,n_steps,n_input])
y = tf.placeholder("float32",[None,n_classes])

#权值与偏置项
weights = {"out_weight":tf.Variable(tf.random_normal([n_hidden,n_classes]))}
biases = {"out_bias":tf.Variable(tf.random_normal([n_classes]))}

def RNN(x,weights,biases):
    """
    RNN结构
    :param x:输入的x值
    :param weights:权重
    :param biases:偏置
    :return:
    """
    x = tf.unstack(x, n_steps, 1)
    lstm_cell = rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)   #LSTM单元
    outputs,states = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)  #获得LSTM的输出
    #这里outputs[-1]取得是最后一次输出？
    return tf.matmul(outputs[-1],weights['out_weight']) + biases['out_bias']   #rnn输出层

train = RNN(x,weights,biases)   #神经网路的训练

#定义损失函数和优化方式
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train,labels=y))  #交叉熵
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#评估模型
correct_train = tf.equal(tf.arg_max(train,1),tf.arg_max(y,1))  #最大值所在的序号是否相等，这里的1是dimension，具体还不知道什么意思
accuracy = tf.reduce_mean(tf.cast(correct_train,tf.float32))  #转成float32是为了浮点数而非整数，reduce_mean是为了取均值

#变量初始化
init = tf.global_variables_initializer()

#在session中运行
with tf.Session() as sess:
    sess.run(init)  #运行初始化变量
    #开始训练
    for i in range(training_iters):
        x_train = x_train.reshape((batch_size,n_steps,n_input))
        sess.run(train,feed_dict={x:x_train,y:y_train})
        if i % display_step == 0:
            #显示当前的准确率
            acc = sess.run(accuracy,feed_dict={x:x_train,y:y_train})
            #当前损失函数
            loss = sess.run(cost, feed_dict={x:x_train,y:y_train})
            print("Iter " + str(i) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
    print("Optimization Finished!")

    #开始测试
    x_test = x_test.reshape((-1, n_steps, n_input))
    print("Testing Accuracy:",sess.run(accuracy, feed_dict={x: x_test, y: y_test}))
    print("Testing Accuracy:",sess.run(train, feed_dict={x: x_test, y: y_test}))


if __name__ == '__main__':
    pass


"""
2621 words in original data but not in word2vec vocabulary
pd dict has 20866 words in total
"""