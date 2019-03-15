# -*- coding: utf-8 -*-
# @Time    : 2018/9/10 19:09
# @Author  : Tianchiyue
# @File    : tan.py
# @Software: PyCharm Community Edition

from sentiment_models.model import BaseModel
from sentiment_models.layers import ConnectAspectLayer, AtaeAttention, LocationAttentionLayer, ClearMaskLayer
from keras.layers import Bidirectional, LSTM, GRU, TimeDistributed, Dense, Flatten, Activation, \
    Lambda, Input, Embedding, dot, concatenate, Dropout, add, RepeatVector, multiply, Conv1D, GlobalMaxPooling1D
import keras.backend as K
import tensorflow as tf


class ARCNN(BaseModel):
    """
    Ref https://github.com/zhouyiwei/tsd/blob/master/model.py
    0.7172,gcae 0.7147,0.7132
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
                               weights=[aspect_embedding_matrix])(self.aspect_input)  # bsz, 1, emb_dims
        aspect_emb = Flatten()(aspect_emb)
        transfered_aspect = Dense(self.config['rnn_output_size'] * 2)(aspect_emb)
        transfered_aspect = RepeatVector(self.config['max_length'])(transfered_aspect)
        transfered_sentence = TimeDistributed(Dense(self.config['rnn_output_size'] * 2))(word_embed)
        merged = add([transfered_sentence, transfered_aspect])
        merged = Activation('tanh')(merged)
        att_alpha = TimeDistributed(Dense(self.config['rnn_output_size'] * 2, activation='sigmoid'))(merged)
        # todo mask 修改merged，修改 transfered 输入时rnn——output
        if self.config['bidirectional']:
            if self.config['rnn'] == 'gru':
                rnn_out = Bidirectional(GRU(self.config['rnn_output_size'],
                                            return_sequences=True,
                                            dropout=self.config['dropout_rate'],
                                            recurrent_dropout=self.config['dropout_rate']))(word_embed)
            else:
                rnn_out = Bidirectional(LSTM(self.config['rnn_output_size'],
                                             return_sequences=True,
                                             dropout=self.config['dropout_rate'],
                                             recurrent_dropout=self.config['dropout_rate']))(word_embed)
        else:
            if self.config['rnn'] == 'gru':
                rnn_out = GRU(self.config['rnn_output_size'],
                              return_sequences=True,
                              dropout=self.config['dropout_rate'],
                              recurrent_dropout=self.config['dropout_rate'])(word_embed)
            else:
                rnn_out = LSTM(self.config['rnn_output_size'],
                               return_sequences=True,
                               dropout=self.config['dropout_rate'],
                               recurrent_dropout=self.config['dropout_rate'])(word_embed)
        att_rnn = multiply([rnn_out, att_alpha])
        convs = []
        att_rnn = ClearMaskLayer()(att_rnn)
        for ksz in self.config['kernel_sizes']:
            x = Conv1D(self.config['filters'], ksz, padding='same', activation='relu')(att_rnn)
            pooling = GlobalMaxPooling1D()(x)
            convs.append(pooling)
        rep = concatenate(convs, axis=-1)
        return rep
