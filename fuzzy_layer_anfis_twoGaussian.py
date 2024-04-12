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


class ANFIS_G_DoublePeak(Layer):
    def __init__(self, n_rules, n_inputs, n_classes, i_fmap, **kwargs):
        super(ANFIS_G_DoublePeak, self).__init__(**kwargs)
        self.n_rules = n_rules
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.index = i_fmap

    def build(self, input_shape):
        # 第一个高斯峰的参数：均值mu1和标准差sigma1
        self.mu1 = self.add_weight(name='mu1',
                                   shape=(1, self.n_inputs),
                                   initializer=equally_spaced_initializer,
                                   trainable=True)
        self.sigma1 = self.add_weight(name='sigma1',
                                      shape=(1, self.n_inputs),
                                      initializer=tf.keras.initializers.RandomUniform(minval=0.7, maxval=1.0, seed=3),
                                      trainable=True)

        # 第二个高斯峰的参数：均值mu2和标准差sigma2
        self.mu2 = self.add_weight(name='mu2',
                                   shape=(1, self.n_inputs),
                                   initializer=equally_spaced_initializer,
                                   trainable=True)
        self.sigma2 = self.add_weight(name='sigma2',
                                      shape=(1, self.n_inputs),
                                      initializer=tf.keras.initializers.RandomUniform(minval=1.0, maxval=1.3, seed=2),
                                      trainable=True)

        # 初始化规则的后件部分参数
        '''self.consequent = self.add_weight(name='consequent',
                                          shape=(self.n_rules, self.n_classes),
                                          initializer=tf.random_normal_initializer(0., 0.1),
                                          trainable=True)'''
        super(ANFIS_G_DoublePeak, self).build(input_shape)

    def call(self, inp):
        inputs = inp[:, :, self.index]
        inputs_expanded1 = K.expand_dims(inputs, 1)
        inputs_expanded2 = K.expand_dims(inputs, 1)
        # 计算第一个高斯峰的隶属函数值
        phi1 = K.exp(-K.square(inputs_expanded1 - self.mu1) / (2 * K.square(self.sigma1)))
        # 计算第二个高斯峰的隶属函数值
        phi2 = K.exp(-K.square(inputs_expanded2 - self.mu2) / (2 * K.square(self.sigma2)))
        # 双峰高斯函数的组合（这里简单地将两个高斯函数的输出相加）
        phi_combined = phi1 + phi2
        phi_combined = tf.reshape(phi_combined, [tf.shape(inp)[0], 8])
        # 计算规则激活度
        '''rule_activation1 = K.prod(phi_combined, axis=2)
        rule_activation_sum1 = K.sum(rule_activation1, axis=1, keepdims=True)
        normalized_activation1 = rule_activation1 / (rule_activation_sum1 + K.epsilon())
        # 应用后件参数
        # weighted_sum = K.sum(normalized_activation1[:, :, None] * self.consequent, axis=1) 
        # 使用Softmax函数将输出转换为概率分布
        # result = tf.nn.softmax(weighted_sum, axis=-1)
        # print('weighted_sum.shape = ', weighted_sum.shape)
        result = normalized_activation1 * inp[:, :, self.index]'''
        return phi_combined

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_classes)

    def get_config(self):
        config = super(ANFIS_G_DoublePeak, self).get_config()
        config.update({
            'n_rules': self.n_rules,
            'n_inputs': self.n_inputs,
            'n_classes': self.n_classes,
            'i_fmap': self.index
        })
        return config

