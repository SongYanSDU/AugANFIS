import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Lambda, Softmax, Multiply, Add, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import random as python_random

seed = 42
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)


def split_tensor(x, num_splits):
    # Assume `x` has shape (batch, height, width, channels)
    # Split the tensor along the channel axis
    return tf.split(x, num_or_size_splits=num_splits, axis=-1)


def fuse_tensors(u1, u2):
    # Implement your fusion strategy here
    # This is just a simple example using concatenation
    return Concatenate(axis=-1)([u1, u2])


def gated_select(u1, u2):
    # Combine u1 and u2 using a softmax gating mechanism
    # Assuming u1 and u2 have the same shape
    gate = Softmax(axis=-1)(Add()([u1, u2]))  # Softmax along the channels
    return Multiply()([gate, u1]), Multiply()([gate, u2])


def gate_cnn(input, nmap, kernel1, kernel2):
    # Define the input
    X_input = input
    # Define the convolutional operations
    U1 = Conv1D(filters=nmap, kernel_size=kernel1, padding='same', activation='relu')(X_input)
    U2 = Conv1D(filters=nmap, kernel_size=kernel2, padding='same', activation='relu')(X_input)
    # Split the tensor into two parts (assuming along the channels axis)
    U1_splits = split_tensor(U1, 2)
    U2_splits = split_tensor(U2, 2)
    # Fuse the split tensors
    fused_U1 = fuse_tensors(U1_splits[0], U2_splits[0])
    fused_U2 = fuse_tensors(U1_splits[1], U2_splits[1])
    # Apply gating and select
    V1, _ = gated_select(fused_U1, fused_U2)
    _, V2 = gated_select(fused_U1, fused_U2)
    # Element-wise sum the gated selections
    V = Add()([V1, V2])
    return V

def gate_fuzzy(U1, U2):
    # U1 = Conv1D(filters=8, kernel_size=3, padding='same', activation='relu')(U1)
    # U2 = Conv1D(filters=8, kernel_size=5, padding='same', activation='relu')(U2)
    # U1 = FuzzyLayer()(x)
    # U2 = FuzzyLayer_dg()(x)
    U1_splits = split_tensor(U1, 2)
    U2_splits = split_tensor(U2, 2)
    # Fuse the split tensors
    fused_U1 = fuse_tensors(U1_splits[0], U2_splits[0])
    fused_U2 = fuse_tensors(U1_splits[1], U2_splits[1])
    # Apply gating and select
    V1, _ = gated_select(fused_U1, fused_U2)
    _, V2 = gated_select(fused_U1, fused_U2)
    # Element-wise sum the gated selections
    V = Add()([V1, V2])
    return V