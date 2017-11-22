 

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf


# tensorflow 中变量的定义。counter 计算器
state = tf.Variable(0, name='counter')
# print(state.name)
# tensorflow 中常量的定义 1
one = tf.constant(1)

# 将变量 state 和常量 one 加起来
new_value = tf.add(state, one)
# new_value 加在 state 上。
update = tf.assign(state, new_value)

# 必须初始化变量
init = tf.initialize_all_variables()  # must have if define variable

with tf.Session() as sess:
    sess.run(init)
    # 做三个循环
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

