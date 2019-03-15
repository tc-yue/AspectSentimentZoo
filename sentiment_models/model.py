# -*- coding: utf-8 -*-
# @Time    : 2018/3/28 15:07
# @Author  : Tianchiyue
# @File    : model.py
# @Software: PyCharm Community Edition
import numpy as np
from keras.layers import Input, Embedding, Dense, Dropout
from keras.models import Model
from keras import backend as K
from keras import optimizers, regularizers, callbacks
from sklearn.metrics import accuracy_score
import logging


class BaseModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.sentence_input = None
        self.aspect_input = None

    def build(self, embedding_matrix, target_embedding_matrix):
        pass

    def compile(self, embedding_matrix,target_embedding_matrix):
        # 文本表示
        rep = self.build(embedding_matrix, target_embedding_matrix)
        if self.config['use_mlp']:
            rep = Dropout(self.config['dropout_rate'])(rep)
            rep = Dense(self.config['hidden_dims'], activation=self.config['activation'])(rep)
        rep = Dropout(self.config['dropout_rate'])(rep)
        if self.config['use_l2']:
            predictions = Dense(self.config['num_classes'],
                                kernel_regularizer=regularizers.l2(self.config['l2']),
                                activation='softmax')(rep)
        else:
            predictions = Dense(self.config['num_classes'],
                                activation='softmax')(rep)
        self.model = Model(inputs=[self.sentence_input, self.aspect_input], outputs=predictions)
        opt = optimizers.get(self.config['optimizer'])
        K.set_value(opt.lr, self.config['learning_rate'])
        self.model.compile(optimizer=opt,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    def predict(self, test_x):
        return self.model.predict(test_x)

    # 根据任务改变
    def evaluate(self, valid_x, valid_y):
        v_pred = [i.argmax() for i in self.predict(valid_x)]
        v_true = [i.argmax() for i in valid_y]
        valid_score = BaseModel.score(v_true, v_pred)
        evaluate_list = self.model.evaluate(valid_x, valid_y, verbose=0)
        return evaluate_list[0], evaluate_list[1], valid_score

    # @staticmethod
    # def batch_iter(data, labels, batch_size, shuffle=True):
    #     num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    def data_generator(self, data, labels, batch_size, num_batches_per_epoch, shuffle=True):
        data_size = len(data[0])
        while True:
            # Shuffle the data at each epoch

            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data0, shuffled_data1 = data[0][shuffle_indices], data[1][shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data0, shuffled_data1 = data[0], data[1]
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X0, X1 = shuffled_data0[start_index: end_index], shuffled_data1[start_index:end_index]
                y = shuffled_labels[start_index: end_index]
                yield [X0, X1], y

        # return num_batches_per_epoch, data_generator()

    def fit(self, train_x, train_y, valid_x, valid_y, predicted=False,filename='trained_models/best.model'):
        lr_decay = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=self.config['lr_decay_epoch'],
                                               min_lr=0.01*self.config['learning_rate'])
        csv_log = callbacks.CSVLogger(filename.replace('.model', '.csv'))
        es = callbacks.EarlyStopping(monitor='val_acc', patience=self.config['n_stop'])
        mc = callbacks.ModelCheckpoint(filename, monitor='val_acc',save_best_only=True,save_weights_only=True)

        train_steps = int((len(train_y) - 1) / self.config['batch_size']) + 1
        valid_steps = int((len(valid_y) - 1) / self.config['batch_size']) + 1
        train_batches = self.data_generator(train_x, train_y, self.config['batch_size'], train_steps)
        valid_batches = self.data_generator(valid_x, valid_y, self.config['batch_size'], valid_steps)
        class_weights = dict(enumerate(train_y.shape[0] / np.bincount(np.argmax(train_y, -1))))
        # class_weights = {0: 2., 1: 1., 2: 1.}
        if self.config['weight_loss']:
            hist = self.model.fit_generator(train_batches, train_steps,
                                            epochs=self.config['epochs'],
                                            callbacks=[lr_decay,csv_log,es,mc],
                                            validation_data=valid_batches,
                                            validation_steps=valid_steps,
                                            class_weight=class_weights)
        else:
            hist = self.model.fit_generator(train_batches, train_steps,
                                            epochs=self.config['epochs'],
                                            callbacks=[lr_decay,csv_log,es,mc],
                                            validation_data=valid_batches,
                                            validation_steps=valid_steps)

        # if self.config['weight_loss']:
        #     hist = self.model.fit(train_x, train_y, batch_size=self.config['batch_size'], epochs=self.config['epochs'],
        #                           validation_data=(valid_x, valid_y), callbacks=[lr_decay, csv_log, es, mc],
        #                           class_weight=class_weights)
        # else:
        #     hist = self.model.fit(train_x, train_y, batch_size=self.config['batch_size'], epochs=self.config['epochs'],
        #                           validation_data=(valid_x, valid_y), callbacks=[lr_decay, csv_log, es, mc])
        best_acc = max(hist.history['val_acc'])
        if predicted:
            self.model.load_weights(filename)
            return self.predict(valid_x), best_acc
        else:
            return best_acc

    @staticmethod
    def score(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

