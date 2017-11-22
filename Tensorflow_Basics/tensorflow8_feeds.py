 

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf

# placeholder 占据一个位置，但是当实际运行的时候需要从外接传进来一个变量
# 输入1
input1 = tf.placeholder(tf.float32)
# 输入2
input2 = tf.placeholder(tf.float32)
# 输出
# 相乘
output = tf.mul(input1, input2)

with tf.Session() as sess:
    # 以 feed_dict 的形式传进去
    # 以字典的形式传进去 input1 和 input2 的值
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
