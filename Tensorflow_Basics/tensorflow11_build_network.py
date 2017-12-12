 

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

##############################################################
# Make up some real data
# 创造真实数据
x_data = np.linspace(-1,1,300)[:, np.newaxis]
# 加入噪声  初始值0  方差0.05 格式 x_data.shape
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

##############################################################
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

##############################################################
# add hidden layer
# 定义隐藏层
# 使用 placeholder 传进来的值
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
# prediction 预计数据
prediction = add_layer(l1, 10, 1, activation_function=None)

##############################################################
# the error between prediciton and real data
# loss 函数，差别。  reduce_sum 求和  tf.square 平方（每个例子的平方）
# 对每一个例子求和，然后对他求一个平均值  rf.reduce_mean
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
# 在误差里面如何学习
# Gradient 梯度   Descent 下降  Optimizer 优化程序
# .minimize（loss） 对误差进行更正
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


##############################################################
# important step
init = tf.initialize_all_variables()  # 初始化
sess = tf.Session() # 定义 Session
sess.run(init) # 设置 session 的指针。进行计算

for i in range(1000):
    # training
    #
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

