# -*- coding: utf-8 -*-
# @Time    : 2018/9/8 20:17
# @Author  : Tianchiyue
# @File    : layers.py
# @Software: PyCharm Community Edition
from keras import backend as K
from keras.engine import Layer
import tensorflow as tf
from keras import initializers, activations
from keras.layers import Dense

class MaskAverageLayer(Layer):
    '''得到评价对象中所有词向量的平均值
    '''

    def __init__(self, keepdims=True, **kwargs):
        self.support_mask = True
        self.keepdims = keepdims
        super(MaskAverageLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        aspect_x = x  # (None*4*300)
        mask_1 = tf.to_float(K.expand_dims(mask, 2))  # None *time_steps*1
        mask_sum = K.sum(tf.to_float(mask_1), axis=1)  # None *1*1
        #         mask(None*timesteps)
        inter = aspect_x * mask_1
        result = K.sum(inter, axis=1) * K.pow(mask_sum, -1)  # None*time_steps
        return K.repeat(result, 1)

    #         return K.expand_dims(K.sum(inter,axis=1)/mask_sum,1)

    def compute_output_shape(self, input_shape):
        if self.keepdims:
            return (input_shape[0], 1, input_shape[2])
        else:
            return (input_shape[0], input_shape[2])

    def compute_mask(self, x, mask=None):
        return None


class ConnectAspectLayer(Layer):
    def __init__(self, **kwargs):
        self.support_mask = True
        super(ConnectAspectLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        left_x = x[0]
        aspect_vector = x[1]  # 32x1*100
        aspect_vector = K.repeat_elements(aspect_vector, K.int_shape(left_x)[1], axis=1)  # 32x24x100
        aspect_vector = aspect_vector * tf.to_float(K.expand_dims(mask[0], 2))

        return K.concatenate([left_x, aspect_vector])

    def compute_output_shape(self, input_shape):

        return (input_shape[0][0], input_shape[0][1], input_shape[0][2] + input_shape[1][2])

    def compute_mask(self, x, mask=None):
        if mask:
            return mask[0]
        else:
            return None


class AtaeAttention(Layer):
    """
    # Input shape
        batch_size,time_steps,hidden_dims
        batch_size,hidden_dims
    """
    def __init__(self,
                 time_steps,
                 activation='tanh',
                 use_bias=False,
                 **kwargs):
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.support_mask = True
        self.time_steps = time_steps
        super(AtaeAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Wh = self.add_weight(name='kernel',
                                 shape=(input_shape[0][2], input_shape[0][2]),
                                 initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 trainable=True)
        self.Wv = self.add_weight(name='vweight',
                                 shape=(input_shape[1][2], input_shape[1][2]),
                                 initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 trainable=True)
        self.w = self.add_weight(name='vweight',
                                 shape=(input_shape[0][2]+input_shape[1][2],1),
                                 initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(1,),
                                        initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                        trainable=True)
        else:
            self.bias = None
        super(AtaeAttention, self).build(input_shape)

    def call(self, x, mask=None):
        h, va = x[0], x[1]
        wh = K.dot(h, self.Wh)
        va = K.repeat_elements(va, self.time_steps, axis=1)
        wv = K.dot(va, self.Wv)
        m = K.concatenate([wh, wv])
        if self.activation is not None:
            m = self.activation(m)
        alpha = K.dot(m, self.w)
        alpha = K.squeeze(alpha, axis=-1)
        alpha = self.softmask(alpha, mask[0])
        return alpha

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0][0], input_shape[0][1]])

    def compute_mask(self, x, mask=None):
        if mask:
            return mask[0]
        else:
            return None

    def softmask(self, x, mask, axis=-1):
        """
        softmax with mask, used in attention mechanism others
        :param x:
        :param mask:
        :param axis:
        :return:
        """
        y = K.exp(x)
        if mask is not None:
            y = y * tf.to_float(mask)
        sumx = K.sum(y, axis=axis, keepdims=True) + 1e-6
        x = y / sumx
        return K.relu(x)


class ClearMaskLayer(Layer):
    """
    after using a layer that supports masking in keras,
    you can use this layer to remove the mask before softmax layer
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(ClearMaskLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, x, mask=None):
        return None
class GetMaskLayer(Layer):
    """
    after using a layer that supports masking in keras,
    you can use this layer to remove the mask before softmax layer
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(GetMaskLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return tf.to_float(mask)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1])

    def compute_mask(self, x, mask=None):
        return mask

class LocationAttentionLayer(Layer):
    """
    # Input shape
        batch_size,time_steps,hidden_dims
    # Output shape
        batch_size,time_steps
    """
    def __init__(self,
                 activation=None,
                 use_bias=True,
                 **kwargs):
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.support_mask = True
        super(LocationAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='kernel',
                                 shape=(input_shape[2], 1),
                                 initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(1,),
                                        initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                        trainable=True)
        else:
            self.bias = None
        super(LocationAttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        output = K.dot(x, self.W)
        # todo match_func concat:vtanh(w[h;t])
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        output = K.batch_flatten(output)
        if self.activation is not None:
            output = self.activation(output)
        atten = self.softmask(output, mask)
        return atten

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0], input_shape[1]])

    def compute_mask(self, x, mask=None):
        if mask is not None:
            return mask
        else:
            return None

    def softmask(self, x, mask, axis=-1):
        """
        softmax with mask, used in attention mechanism others
        :param x:
        :param mask:
        :param axis:
        :return:
        """
        y = K.exp(x)
        if mask is not None:
            y = y * tf.to_float(mask)
        sumx = K.sum(y, axis=axis, keepdims=True) + 1e-6
        x = y / sumx
        return K.relu(x)
