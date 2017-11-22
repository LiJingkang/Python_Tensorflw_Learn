 

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
import tensorflow as tf

# 添加一层神经元
# add_layer(输入，输入格式，输出格式，激活函数)
def add_layer(inputs, in_size, out_size, activation_function=None ):
    # 定义突触权值  格式定义（有可能为一个矩阵）
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 定义偏移量（列表）（1行，out_size 列）—— 推荐值不为 0
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 定义 输出 = 输入 * 权值 + 偏移量
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    # 如果激活函数不存在，直接当作输出
    if activation_function is None:
        outputs = Wx_plus_b
    # 激活函数存在，对计算数据进行处理。
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
