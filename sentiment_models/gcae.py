# -*- coding: utf-8 -*-
# @Time    : 2018/7/26 21:40
# @Author  : Tianchiyue
# @File    : cnn.py.py
# @Software: PyCharm Community Edition

from sentiment_models.model import BaseModel
from keras.layers import Conv1D, GlobalMaxPooling1D, concatenate, BatchNormalization, Activation, PReLU, add, \
    MaxPooling1D, SpatialDropout1D, Dense, Input, Embedding, multiply, Lambda
from keras import initializers, regularizers, constraints, callbacks
from sentiment_models.layers import ClearMaskLayer, GetMaskLayer
import keras.backend as K


class GCAE(BaseModel):
    def build(self, embedding_matrix, aspect_embedding_matrix):
        self.sentence_input = Input(shape=(self.config['max_length'],),
                                    dtype='int32',
                                    name='sentence_input')
        self.aspect_input = Input((1,),
                                  dtype='int32',
                                  name='aspect_input')
        word_embed = Embedding(embedding_matrix.shape[0],
                               embedding_matrix.shape[1],
                               trainable=self.config['embed_trainable'],
                               weights=[embedding_matrix],
                               mask_zero=True
                               )(self.sentence_input)  # bsz, time_steps, emb_dims
        word_mask = GetMaskLayer()(word_embed)
        word_mask = Lambda(lambda xx: K.expand_dims(xx, axis=-1))(word_mask)
        word_embed = ClearMaskLayer()(word_embed)
        aspect_emb = Embedding(aspect_embedding_matrix.shape[0],
                               aspect_embedding_matrix.shape[1],
                               trainable=self.config['aspect_emb_trainable'],
                               weights=[aspect_embedding_matrix])(self.aspect_input)
        convs_x = []
        convs_y = []
        for ksz in self.config['kernel_sizes']:
            x = Conv1D(self.config['filters'], ksz, padding='same', activation='tanh',name='tanh_conv{}'.format(ksz))(word_embed)
            y = Conv1D(self.config['filters'], ksz, padding='same',name='relu_conv{}'.format(ksz))(word_embed)
            fc = Dense(self.config['filters'], use_bias=False, name='fc_{}'.format(ksz))(aspect_emb)
            y = add([y, fc])
            y = Activation(activation='relu')(y)
            convs_x.append(x)
            convs_y.append(y)
        x = [multiply([i, j]) for i, j in zip(convs_x, convs_y)]
        mask_x = [multiply([i, word_mask]) for i in x]
        x0 = [GlobalMaxPooling1D()(i) for i in mask_x]

        merged = concatenate(x0, axis=-1)
        return merged


# class GCAE(BaseModel):
#
#     def build(self, embedding_matrix, aspect_embedding_matrix):
#         self.sentence_input = Input(shape=(self.config['max_length'],),
#                                     dtype='int32',
#                                     name='sentence_input')
#         self.aspect_input = Input((1,),
#                              dtype='int32',
#                              name='aspect_input')
#         word_embed = Embedding(embedding_matrix.shape[0],
#                                embedding_matrix.shape[1],
#                                trainable=self.config['embed_trainable'],
#                                weights=[embedding_matrix],
#                                # mask_zero=True
#                                )(self.sentence_input)  # bsz, time_steps, emb_dims
#         aspect_emb = Embedding(aspect_embedding_matrix.shape[0],
#                                aspect_embedding_matrix.shape[1],
#                                trainable=self.config['aspect_emb_trainable'],
#                                weights=[aspect_embedding_matrix])(self.aspect_input)
#         convs_x = []
#         convs_y = []
#         for ksz in self.config['kernel_sizes']:
#             x = Conv1D(self.config['filters'], ksz, padding='same', activation='tanh')(word_embed)
#             y = Conv1D(self.config['filters'], ksz, padding='same', activation='relu')(word_embed)
#             fc = Dense(self.config['filters'], use_bias=False)(aspect_emb)
#             y = add([y, fc])
#             convs_x.append(x)
#             convs_y.append(y)
#         x = [multiply([i,j]) for i, j in zip(convs_x,convs_y)]
#
#         x0 = [GlobalMaxPooling1D()(i) for i in x]
#
#         merged = concatenate(x0, axis=-1)
#         return merged
