import numpy as np
import tensorflow as tf
import random as python_random

seed = 42
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)


class FourierMixStyle(tf.keras.layers.Layer):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def mixup(self, x, y):
        B = tf.shape(x)[0]
        # Use TensorFlow operations to generate beta-distributed random numbers
        lmda1 = tf.random.uniform(shape=(B, 1, 1), minval=0, maxval=1, dtype=x.dtype)
        lmda1 = tf.math.pow(lmda1, 1.0 / self.alpha)  # Adjust this formula as needed for your beta distribution
        lmda2 = tf.random.uniform(shape=(B, 1, 1), minval=0, maxval=1, dtype=x.dtype)
        lmda2 = tf.math.pow(lmda2, 1.0 / self.alpha)
        # tf.print(x.shape)
        # 傅里叶变换
        x_fft = tf.signal.fft(tf.cast(x, tf.complex64))
        y_fft = tf.signal.fft(tf.cast(y, tf.complex64))

        # 提取实部和虚部
        x_real, x_imag = tf.math.real(x_fft), tf.math.imag(x_fft)
        y_real, y_imag = tf.math.real(y_fft), tf.math.imag(y_fft)

        real_mixed = x_real * lmda1 + y_real * (1 - lmda1)
        imag_mixed = x_imag * lmda2 + y_imag * (1 - lmda2)
        # 重新组合实部和虚部，进行傅里叶反变换
        x_fft_mix = tf.complex(real_mixed, imag_mixed)
        x_ifft = tf.signal.ifft(x_fft_mix)
        return tf.cast(tf.math.real(x_ifft), tf.float32)

    def call(self, inputs):
        x, y = inputs
        mixed_x = self.mixup(x, y)
        mixed_y = self.mixup(y, x)
        return mixed_x, mixed_y

    def get_config(self):
        config = super(FourierMixStyle, self).get_config()
        config.update({
            'p': self.p,
            'alpha': self.alpha,
            'eps': self.eps
        })
        return config