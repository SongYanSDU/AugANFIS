from tensorflow.keras.layers import Layer
import tensorflow as tf
import tensorflow.keras.backend as K

import numpy as np
import random as python_random

seed = 42
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)


# Custom weight initializer
def equally_spaced_initializer(shape, minval=-1.5, maxval=1.5, dtype=tf.float32):
    """
    Custom weight initializer:
        equally spaced weights along an operating range of [minval, maxval].
    """
    linspace = tf.reshape(tf.linspace(minval, maxval, shape[0]),(-1, 1))
    return tf.Variable(tf.tile(linspace, (1, shape[1])))


class ANFIS_G(Layer):
    def __init__(self, n_rules, n_inputs, n_classes, i_fmap, **kwargs):
        super(ANFIS_G, self).__init__(**kwargs)
        self.n_rules = n_rules
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.index = i_fmap

    def build(self, input_shape):
        # 初始化高斯隶属函数的参数：均值mu和标准差sigma
        self.mu = self.add_weight(name='mu',
                                  shape=(1, self.n_inputs),
                                  initializer=equally_spaced_initializer,
                                  trainable=True)
        self.sigma = self.add_weight(name='sigma',
                                     shape=(1, self.n_inputs),
                                     initializer=tf.keras.initializers.RandomUniform(
                                             minval=.7, maxval=1.3, seed=1),
                                     trainable=True)
        # 初始化规则的后件部分参数
        '''self.consequent = self.add_weight(name='consequent',
                                          shape=(self.n_rules, self.n_classes),
                                          initializer=tf.random_normal_initializer(0., 0.1),
                                          trainable=True)'''
        super(ANFIS_G, self).build(input_shape)

    def call(self, inp):
        # 计算每个规则的高斯隶属函数值
        inputs = inp[:, :, self.index]
        # print('normalized_activation=', inputs.shape)
        inputs_expanded = K.expand_dims(inputs, 1)
        mu_expanded = K.expand_dims(self.mu, 0)
        sigma_expanded = K.expand_dims(self.sigma, 0)
        phi = K.exp(-K.square(inputs_expanded - mu_expanded) / (2 * K.square(sigma_expanded)))
        phi = tf.reshape(phi,[tf.shape(inp)[0], 8])
        # 计算规则激活度
        ''' rule_activation = K.prod(phi, axis=2)

        # 规则归一化
        rule_activation_sum = K.sum(rule_activation, axis=1, keepdims=True)
        normalized_activation = rule_activation / (rule_activation_sum + K.epsilon())
        # Perform element-wise multiplication with broadcasting
        result = normalized_activation * inp[:, :, self.index]'''
        # tf.print('normalized_activation=', normalized_activation)
        # 应用后件参数
        # consequent_expanded = K.expand_dims(self.consequent, 0)
        # weighted_sum = K.sum(normalized_activation[:, :, None] * consequent_expanded, axis=1)
        # tf.print('weighted_sum=', weighted_sum.shape)
        # 使用Softmax函数将输出转换为概率分布
        # result = tf.nn.softmax(weighted_sum, axis=-1)
        # print('normalized_activation=', normalized_activation.shape)
        return phi

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_classes)

    def get_config(self):
        config = super(ANFIS_G, self).get_config()
        config.update({
            'n_rules': self.n_rules,
            'n_inputs': self.n_inputs,
            'n_classes': self.n_classes,
            'i_fmap': self.index
        })
        return config