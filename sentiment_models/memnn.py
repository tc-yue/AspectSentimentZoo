# -*- coding: utf-8 -*-
# @Time    : 2018/9/15 16:36
# @Author  : Tianchiyue
# @File    : memnn.py
# @Software: PyCharm Community Edition

from sentiment_models.model import BaseModel
from sentiment_models.layers import ConnectAspectLayer, AtaeAttention
from keras.layers import Bidirectional, LSTM, GRU, TimeDistributed, Dense, Flatten, Activation, \
    Lambda, Input, Embedding, dot, concatenate,Dropout,add
import keras.backend as K
import tensorflow as tf


class MEMNN(BaseModel):
    """
    Ref:https://github.com/tc-yue/DeepLearningPractice2017/blob/master/AspectSentimentAnalyze/car_review_memory_nn.ipynb
    """

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
        aspect_emb = Embedding(aspect_embedding_matrix.shape[0],
                               aspect_embedding_matrix.shape[1],
                               trainable=self.config['aspect_emb_trainable'],
                               weights=[aspect_embedding_matrix])(self.aspect_input)    # bsz, 1, emb_dims
        if self.config['connect_aspect']:
            input_matrix = ConnectAspectLayer()([word_embed, aspect_emb])
        else:
            input_matrix = word_embed
        if self.config['bidirectional']:
            if self.config['rnn'] == 'gru':
                rnn_out = Bidirectional(GRU(self.config['rnn_output_size'],
                                            return_sequences=True,
                                            dropout=self.config['dropout_rate'],
                                            recurrent_dropout=self.config['dropout_rate']))(input_matrix)
            else:
                rnn_out = Bidirectional(LSTM(self.config['rnn_output_size'],
                                             return_sequences=True,
                                             dropout=self.config['dropout_rate'],
                                             recurrent_dropout=self.config['dropout_rate']))(input_matrix)
        else:
            if self.config['rnn'] == 'gru':
                rnn_out = GRU(self.config['rnn_output_size'],
                              return_sequences=True,
                              dropout=self.config['dropout_rate'],
                              recurrent_dropout=self.config['dropout_rate'])(input_matrix)
            else:
                rnn_out = LSTM(self.config['rnn_output_size'],
                               return_sequences=True,
                               dropout=self.config['dropout_rate'],
                               recurrent_dropout=self.config['dropout_rate'])(input_matrix)
        attention_alpha = AtaeAttention(time_steps=self.config['max_length'])([rnn_out, aspect_emb])
        r = dot([attention_alpha, rnn_out], axes=[1,1], name='attention_mul')
        # r = Flatten()(r)
        r = Dropout(self.config['dropout_rate'])(r)
        r = Dense(self.config['rnn_output_size'],use_bias=False)(r)
        h = Lambda(lambda x: tf.slice(x, [0, self.config['max_length'] - 1, 0], [-1, 1, -1]))(rnn_out)
        h = Flatten()(h)
        h = Dense(self.config['rnn_output_size'], use_bias=False)(h)
        h_star = add([r,h])
        h_star = Activation('tanh')(h_star)
        return h_star