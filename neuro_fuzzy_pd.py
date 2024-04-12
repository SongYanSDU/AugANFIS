import tensorflow as tf
from scipy.io import loadmat
from tensorflow.python import keras
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense, concatenate, Layer,
                                     GaussianNoise, Reshape, GlobalMaxPooling1D, BatchNormalization)
from tensorflow.keras.models import Model
from tqdm import tqdm
import gc
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import pydot
import graphviz
from sklearn.model_selection import KFold
from scipy.signal import butter, lfilter
import numpy as np
from numpy import linalg as LA
from itertools import product
import scipy
import random
import os
import gc
import matplotlib.pyplot as plt
from fuzzy_layer_anfis_Gaussian import ANFIS_G
from fuzzy_layer_anfis_twoGaussian import ANFIS_G_DoublePeak
from mix_style import FourierMixStyle
import random as python_random
import scipy.io as sio
from gated_cnn import gate_cnn

tf.config.experimental_run_functions_eagerly(True)
seed = 42
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)

nclasses = 3
N_map = 16


def normalized_act(inp1, inp2, inp3, inp4, x):
    # 计算规则激活度
    rule_activation1 = K.prod(inp1, axis=2)
    rule_activation2 = K.prod(inp2, axis=2)
    rule_activation3 = K.prod(inp3, axis=2)
    rule_activation4 = K.prod(inp4, axis=2)
    # 规则归一化
    # rule_activation_sum = K.sum(rule_activation, axis=1, keepdims=True)
    rule_activation_sum = K.sum(rule_activation1 + rule_activation2 + rule_activation3 + rule_activation4, axis=1, keepdims=True)
    normalized_activation1 = rule_activation1 / (rule_activation_sum + K.epsilon())
    normalized_activation2 = rule_activation2 / (rule_activation_sum + K.epsilon())
    normalized_activation3 = rule_activation3 / (rule_activation_sum + K.epsilon())
    normalized_activation4 = rule_activation4 / (rule_activation_sum + K.epsilon())
    normalized_activation_expanded1 = K.expand_dims(normalized_activation1, axis=2)
    normalized_activation_expanded2 = K.expand_dims(normalized_activation2, axis=2)
    normalized_activation_expanded3 = K.expand_dims(normalized_activation3, axis=2)
    normalized_activation_expanded4 = K.expand_dims(normalized_activation4, axis=2)
    # Perform element-wise multiplication with broadcasting
    result1 = normalized_activation_expanded1 * x
    result2 = normalized_activation_expanded2 * x
    result3 = normalized_activation_expanded3 * x
    result4 = normalized_activation_expanded4 * x
    # print(result.shape)
    return result1, result2, result3, result4


def pre_acc(test_data, test_label, model):
    test_r = model.predict(test_data)
    ypred = test_r
    ypred = np.argmax(ypred, axis=1)
    ypred = ypred.reshape(len(ypred), )
    dd = np.argwhere(ypred == test_label)
    dd = np.array(dd)
    acc = float(len(dd)) / float(test_label.shape[0])
    return acc


class AdjustLRCallback(tf.keras.callbacks.Callback):
    def __init__(self, learning_rate, gammas, schedule):
        super().__init__()
        self.learning_rate = learning_rate
        self.gammas = gammas
        self.schedule = schedule
        assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"

    def on_epoch_end(self, epoch, logs=None):
        lr = self.learning_rate
        for gamma, step in zip(self.gammas, self.schedule):
            if epoch >= step:
                lr *= gamma
            else:
                break
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        print(f"Learning Rate adjusted to: {lr}")


def cnn(inp, n_map=N_map):
    x = gate_cnn(inp, N_map, 17, 3)
    x = MaxPooling1D(16)(x)
    x = gate_cnn(x, N_map, 17, 3)
    x = MaxPooling1D(16)(x)
    x = Conv1D(n_map, 3, padding='same', activation='relu')(x)
    x = MaxPooling1D(2)(x)
    return x


def fuzzy(x):
    fuzzy_inference_1 = []
    fuzzy_inference_2 = []
    for i in tqdm(range(N_map)):
        f_block = ANFIS_G(n_rules=1, n_inputs=8, n_classes=nclasses, i_fmap=i)(x)
        fuzzy_inference_1.append(f_block)
        f_block = ANFIS_G_DoublePeak(n_rules=1, n_inputs=8, n_classes=nclasses, i_fmap=i)(x)
        fuzzy_inference_2.append(f_block)
    # 将fuzzy_inference中的矩阵堆叠成一个新的张量
    fuzzy_stacked_1 = tf.stack(fuzzy_inference_1)
    fuzzy_stacked_2 = tf.stack(fuzzy_inference_2)
    # 调整fuzzy_inference_stacked的形状
    fuzzy_stacked_1 = tf.transpose(fuzzy_stacked_1, [1, 2, 0])
    fuzzy_stacked_2 = tf.transpose(fuzzy_stacked_2, [1, 2, 0])
    return fuzzy_stacked_1, fuzzy_stacked_2


"""Implement Fuzzy CNN 
define fuzzy CNN architecture"""
def fcnn(signal_length=4096, n_channels=1, num_classes=nclasses, n_map=N_map):
    inp = Input(shape=[signal_length, n_channels])
    x = cnn(inp, n_map=n_map)
    x1, x2 = fuzzy(x)
    mixed_x, mixed_y = FourierMixStyle()([x1, x2])
    x1, x2, mixed_x, mixed_y = normalized_act(x1, x2, mixed_x, mixed_y, x)

    fuzzy_stacked_1 = Flatten()(x1)
    fuzzy_stacked_2 = Flatten()(x2)
    mixed_x = Flatten()(mixed_x)
    mixed_y = Flatten()(mixed_y)

    fuzzy_stacked = concatenate([fuzzy_stacked_1, fuzzy_stacked_2, mixed_x, mixed_y], axis=1)
    fuzzy_stacked = Dense(128, name='dense_layer')(fuzzy_stacked)
    out = Dense(num_classes, 'softmax')(fuzzy_stacked)
    model = Model(inputs=inp, outputs=out)
    return model


from data_loader import Paderborn_dataset, Lboro_dataset
data1, label1, data2, label2, data3, label3, data4, label4 = Paderborn_dataset()

train_features = data2
train_labels = label2
# 使用示例
initial_learning_rate = 0.001
gammas = [0.1, 0.1]  # 乘以0.1表示每次衰减到原来的10%
schedule = [10, 20]  # 每30个epoch衰减一次
adjust_lr_callback = AdjustLRCallback(initial_learning_rate, gammas, schedule)
# optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate, momentum=0.9, decay=0.0005, nesterov=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
"""Show Model"""
model = fcnn(signal_length=4096, n_channels=1, num_classes=nclasses, n_map=N_map)
model.summary()

num_epochs = 30
# model.compile(optimizer=optimizer, loss=global_loss)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
# 假设 `model` 是你的原始模型，`my_layer` 是你想要输出的层的名称
layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('dense_layer').output)
for i in range(1):
    print('*********************************************************************************************************')
    history = model.fit(train_features, train_labels, batch_size=512, epochs=num_epochs,
                         verbose=2, callbacks=[adjust_lr_callback])
    # history = model.fit(train_features, train_labels, batch_size=512,
    #                   epochs=num_epochs, verbose=2, callbacks=[adjust_lr_callback])
    # Generate generalization metrics
    acc1 = pre_acc(data1, label1, model)
    print('acc1 = ', acc1)
    acc2 = pre_acc(data3, label3, model)
    print('acc2 = ', acc2)
    acc3 = pre_acc(data4, label4, model)
    print('acc3 = ', acc3)

# 使用训练好的模型来提取特征
'''train_features = layer_model.predict(train_features)  # 或任何你希望提取特征的数据
test_features = layer_model.predict(data1)  # 或任何你希望提取特征的数据
# 使用原始模型来获取预测结果
test_predictions = model.predict(data1)  # 或任何你希望获取预测结果的数据
test_predictions = np.argmax(test_predictions[:, :], axis=-1)
test_labels = label1

path = 'train4'
sio.savemat('./results/' + path + '.mat',
                        {'data1_features': train_features,
                                'train_labels': train_labels,
                                'data4_features': test_features,
                                'data4_predictions': test_predictions,
                                'label4': test_labels})
# 假设你的模型名为 `model`
model.save('./results/' + path + '.h5')'''