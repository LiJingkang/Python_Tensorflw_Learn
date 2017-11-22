 

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf
import numpy as np  # 科学计算模块

# create data
# 输入 x
x_data = np.random.rand(100).astype(np.float32) # 生成100个随机数列，数据类型为 float32
# 输出 y
y_data = x_data*0.1 + 0.3 # 预测的data，Weight接近0.1, Biases 接近0.3

### create tensorflow structure start ###
# 学习过程，从初始值不断学习的过程
# 开始创建结构
# 突触权值 w
# 使用tf变量，用tf的随机变量生成参数。结构是[1]一维。生成值范围 -1.0 到 1.0
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# 偏置 b
# 生成的初始值为 0
biases = tf.Variable(tf.zeros([1]))

# 预测的y。
# 会提升 y 的准确度
y = Weights*x_data + biases
# loss 函数
# 预测y 与实际y 的区别。
loss = tf.reduce_mean(tf.square(y-y_data))
# 激活函数
# 通过激活函数来优化 w
# 优化器 optimizer。 0.5 学习效率
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 训练。  每一步训练使用优化器减少它的误差
train = optimizer.minimize(loss)

# 在神经网络中初始化所有的变量。
init = tf.initialize_all_variables()
### create tensorflow structure end ###


# 定义Session
sess = tf.Session()
# 运行，处理初始化好的init
sess.run(init)          # Very important

for step in range(201):
    # 用session的指针指向train。开始训练
    sess.run(train)
    if step % 20 == 0:
        # 打印目前的权值 和
        print(step, sess.run(Weights), sess.run(biases))


