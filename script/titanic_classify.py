#!/usr/bin/python
# -*- coding: gbk -*-

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("../data/train.csv")
data.info()

#�滻�ֶ��е�ֵ
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == "male" else 0)
mean_age = data['Age'].mean()
data['Age'][data.Age.isnull()] = mean_age
#��ȱʧֵ���Ϊ0
data = data.fillna(0)
#��ȡ���ݵ���Щ�У���Ϊ��������
dataset_X = data[["Sex", "Age", "Pclass", "SibSp", "Parch", 'Fare']]
dataset_X = dataset_X.as_matrix()
#������λ�������
data['Decreased'] = data['Survived'].apply(lambda s: int(not s))
dataset_Y = data[['Decreased', 'Survived']]
dataset_Y = dataset_Y.as_matrix()
#���ݽ����з�
X_train, X_test, Y_train, Y_test = train_test_split( \
        dataset_X, dataset_Y, test_size=0.2, random_state=42)
#������������ռλ��
#shape�����еĵ�һ��Ԫ��ΪNone����ʾ����ͬʱ������������¼,һ����minbatch�������ڶ���Ԫ�������ݵ�ά��
X = tf.placeholder(tf.float32, shape=[None, 6], name='input_data')
Y = tf.placeholder(tf.float32, shape=[None, 2], name='outoput_data')
W = tf.Variable(tf.random_normal([6, 2]), name='weight')
b = tf.Variable(tf.zeros([2]), name="bias")
y_pred = tf.nn.softmax(tf.matmul(X, W) + b)
#ʹ�ý������������Ż�����
cross_entropy = -tf.reduce_sum(Y * tf.log(y_pred + 1e-10), reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)
#sdg�Ż�����
train_op = tf.train.GradientDescentOptimizer(0.0003).minimize(cost)
saver = tf.train.Saver()
with tf.Session() as sess:
    #��ʼ������
    tf.global_variables_initializer().run()
    #��ʼѵ��
    for epoch in range(300):
        total_loss = 0.0
        for i in range(len(X_train)):
            feed = {X: [X_train[i]], Y: [Y_train[i]]}
            _, loss = sess.run([train_op, cost], feed_dict = feed)
            total_loss += loss
        print 'Epoch: %04d, total loss =%.09f' % (epoch + 1, total_loss)
    print "training complete "
    save_path = saver.save(sess, "/Users/lixuejian02/work_space/tensorflow_learning/script/model.ckpt")
    #��ʼ����
    pred = sess.run(y_pred, feed_dict = {X: X_test})
    correct = np.equal(np.argmax(pred, 1), np.argmax(Y_test, 1))
    accuracy = np.mean(correct.astype(np.float32))
    print "Accuracy on validation dataset: %.9f" % accuracy

data = pd.read_csv("../data/test.csv")
#�滻�ֶ��е�ֵ
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == "male" else 0)
data['Age'][data.Age.isnull()] = mean_age
#��ȱʧֵ���Ϊ0
data = data.fillna(0)
#��ȡ���ݵ���Щ�У���Ϊ��������
test_X = data[["Sex", "Age", "Pclass", "SibSp", "Parch", 'Fare']]
with tf.Session() as sess:
    #����ģ���ĵ�
    saver.restore(sess, "model.ckpt")
    #���򴫲�����
    predictions = np.argmax(sess.run(y_pred, feed_dict = {X: test_X}), 1)    
    #���ɽ��
    submission = pd.DataFrame({
        "PassengerId": data["PassengerId"],
        "Survived":predictions
        })
    submission.to_csv("titanic_submission.csv", index = False)


