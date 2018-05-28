#!/usr/bin/python
# -*- coding: gbk -*-

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("../data/train.csv")
data.info()

#替换字段中的值
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == "male" else 0)
mean_age = data['Age'].mean()
data['Age'][data.Age.isnull()] = mean_age
#将缺失值填充为0
data = data.fillna(0)
#获取数据的这些列，作为输入特征
dataset_X = data[["Sex", "Age", "Pclass", "SibSp", "Parch", 'Fare']]
dataset_X = dataset_X.as_matrix()
#构造两位输出数据
data['Decreased'] = data['Survived'].apply(lambda s: int(not s))
dataset_Y = data[['Decreased', 'Survived']]
dataset_Y = dataset_Y.as_matrix()
#数据进行切分
X_train, X_test, Y_train, Y_test = train_test_split( \
        dataset_X, dataset_Y, test_size=0.2, random_state=42)
#声明输入数据占位符
#shape参数中的第一个元素为None，表示可以同时放入任意条记录,一般是minbatch参数，第二个元素是数据的维度
X = tf.placeholder(tf.float32, shape=[None, 6], name='input_data')
Y = tf.placeholder(tf.float32, shape=[None, 2], name='outoput_data')
W = tf.Variable(tf.random_normal([6, 2]), name='weight')
b = tf.Variable(tf.zeros([2]), name="bias")
y_pred = tf.nn.softmax(tf.matmul(X, W) + b)
#使用交叉熵来评估优化方向
cross_entropy = -tf.reduce_sum(Y * tf.log(y_pred + 1e-10), reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)
#sdg优化算子
train_op = tf.train.GradientDescentOptimizer(0.0003).minimize(cost)
saver = tf.train.Saver()
with tf.Session() as sess:
    #初始化变量
    tf.global_variables_initializer().run()
    #开始训练
    for epoch in range(300):
        total_loss = 0.0
        for i in range(len(X_train)):
            feed = {X: [X_train[i]], Y: [Y_train[i]]}
            _, loss = sess.run([train_op, cost], feed_dict = feed)
            total_loss += loss
        print 'Epoch: %04d, total loss =%.09f' % (epoch + 1, total_loss)
    print "training complete "
    save_path = saver.save(sess, "/Users/lixuejian02/work_space/tensorflow_learning/script/model.ckpt")
    #开始测试
    pred = sess.run(y_pred, feed_dict = {X: X_test})
    correct = np.equal(np.argmax(pred, 1), np.argmax(Y_test, 1))
    accuracy = np.mean(correct.astype(np.float32))
    print "Accuracy on validation dataset: %.9f" % accuracy

data = pd.read_csv("../data/test.csv")
#替换字段中的值
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == "male" else 0)
data['Age'][data.Age.isnull()] = mean_age
#将缺失值填充为0
data = data.fillna(0)
#获取数据的这些列，作为输入特征
test_X = data[["Sex", "Age", "Pclass", "SibSp", "Parch", 'Fare']]
with tf.Session() as sess:
    #加载模型文档
    saver.restore(sess, "model.ckpt")
    #正向传播计算
    predictions = np.argmax(sess.run(y_pred, feed_dict = {X: test_X}), 1)    
    #生成结果
    submission = pd.DataFrame({
        "PassengerId": data["PassengerId"],
        "Survived":predictions
        })
    submission.to_csv("titanic_submission.csv", index = False)


