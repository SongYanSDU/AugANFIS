# -- coding: utf-8 --
import numpy as np
import scipy.io as sio
import tensorflow as tf
import random as python_random

seed = 42
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)

def data_shuffle(data1, label1):
    index1 = np.arange(np.size(data1, 0))
    np.random.shuffle(index1)
    data1 = data1[index1, :]
    label1 = label1[index1, ]
    # label1 = label1.reshape(len(label1), 1)
    return data1, label1


def Paderborn_dataset():
    pd1 = sio.loadmat('E:\\python-code\\dataset\\paderborn\\N15_M07_F10.mat')
    data1 = pd1['N15_M07_F10'][:, 1:]
    label1 = pd1['N15_M07_F10'][:, 0]
    data1, label1 = data_shuffle(data1, label1)

    pd2 = sio.loadmat('E:\\python-code\\dataset\\paderborn\\N15_M01_F10.mat')
    data2 = pd2['N15_M01_F10'][:, 1:]
    label2 = pd2['N15_M01_F10'][:, 0]
    data2, label2 = data_shuffle(data2, label2)

    pd3 = sio.loadmat('E:\\python-code\\dataset\\paderborn\\N15_M07_F04.mat')
    data3 = pd3['N15_M07_F04'][:, 1:]
    label3 = pd3['N15_M07_F04'][:, 0]
    data3, label3 = data_shuffle(data3, label3)

    pd4 = sio.loadmat('E:\\python-code\\dataset\\paderborn\\N09_M07_F10.mat')
    data4 = pd4['N09_M07_F10'][:, 1:]
    label4 = pd4['N09_M07_F10'][:, 0]
    data4, label4 = data_shuffle(data4, label4)
    return data1, label1, data2, label2, data3, label3, data4, label4


def Lboro_dataset():
    pd1 = sio.loadmat('E:\\python-code\\dataset\\loughborough\\acceleration\\data_2000.mat')
    data1 = pd1['data_2000'][:, 1:]
    label1 = pd1['data_2000'][:, 0]
    data1, label1 = data_shuffle(data1, label1)

    pd2 = sio.loadmat('E:\\python-code\\dataset\\loughborough\\acceleration\\data_900_1500.mat')
    data2 = pd2['data_900_1500'][:, 1:]
    label2 = pd2['data_900_1500'][:, 0]
    data2, label2 = data_shuffle(data2, label2)

    pd3 = sio.loadmat('E:\\python-code\\dataset\\loughborough\\acceleration\\data_1500_2000.mat')
    data3 = pd3['data_1500_2000'][:, 1:]
    label3 = pd3['data_1500_2000'][:, 0]
    data3, label3 = data_shuffle(data3, label3)

    pd4 = sio.loadmat('E:\\python-code\\dataset\\loughborough\\acceleration\\data_900_2000.mat')
    data4 = pd4['data_900_2000'][:, 1:]
    label4 = pd4['data_900_2000'][:, 0]
    data4, label4 = data_shuffle(data4, label4)
    return data1, label1, data2, label2, data3, label3, data4, label4


def Lboro_dataset_ac():
    pd1 = sio.loadmat('E:\\python-code\\dataset\\loughborough\\acoustic\\data_2000.mat')
    data1 = pd1['data_2000'][:, 1:]
    label1 = pd1['data_2000'][:, 0]
    data1, label1 = data_shuffle(data1, label1)

    pd2 = sio.loadmat('E:\\python-code\\dataset\\loughborough\\acoustic\\data_900_1500.mat')
    data2 = pd2['data_900_1500'][:, 1:]
    label2 = pd2['data_900_1500'][:, 0]
    data2, label2 = data_shuffle(data2, label2)

    pd3 = sio.loadmat('E:\\python-code\\dataset\\loughborough\\acoustic\\data_1500_2000.mat')
    data3 = pd3['data_1500_2000'][:, 1:]
    label3 = pd3['data_1500_2000'][:, 0]
    data3, label3 = data_shuffle(data3, label3)

    pd4 = sio.loadmat('E:\\python-code\\dataset\\loughborough\\acoustic\\data_900_2000.mat')
    data4 = pd4['data_900_2000'][:, 1:]
    label4 = pd4['data_900_2000'][:, 0]
    data4, label4 = data_shuffle(data4, label4)
    return data1, label1, data2, label2, data3, label3, data4, label4
