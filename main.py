# -*- coding: utf-8 -*-
# @Time    : 2018/9/9 20:20
# @Author  : Tianchiyue
# @File    : sentiment_clf.py
# @Software: PyCharm Community Edition


from sentiment_models import configs, atae, tan, gcae, arcnn
import numpy as np
import pickle
from keras.backend.tensorflow_backend import set_session
import os
import random as rn
import tensorflow as tf
import keras.backend as K
import logging
import sys
import argparse
from collections import Counter
from sklearn.model_selection import StratifiedKFold
import itertools
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

def init_env(gpu_id):
    """
    设置gpuid
    :param gpu_id:字符串
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    set_session(tf.Session(config=tf_config))
    logging.info('GPU%s ready!' % gpu_id)


def rand_set():
    # 设置随机种子
    os.environ['PYTHONHASHSEED'] = '7'
    np.random.seed(7)
    rn.seed(7)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(7)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    logging.info('\t=======Init Over=======')


def load_data(data_path):
    with open(data_path, 'rb')as f:
        data = pickle.load(f)
    logging.info('\t=======Data Loaded=======')
    return data


def load_config(model_name):
    """
    加载config，dict 是可变对象，传入引用，如果在函数内部改变，对象原始值也改变
    Ref:http://www.runoob.com/w3cnote/python-understanding-dict-copy-shallow-or-deep.html
        https://www.cnblogs.com/loleina/p/5276918.html
    """
    if model_name in ['atae']:
        config = configs.atae_config
    elif model_name in ['tan']:
        config = configs.tan_config
    elif model_name in ['gcae']:
        config = configs.gcae_config
    elif model_name in ['arcnn']:
        config = configs.arcnn_config
    else:
        return None
    logging.info('===={}配置文件加载完毕===='.format(model_name))
    return config


def load_model(model_name, model_config, word_embedding_matrix, aspect_embedding_matrix):
    if model_name == 'atae':
        model = atae.ATAE(model_config)
    elif model_name == 'tan':
        model = tan.TAN(model_config)
    elif model_name == 'gcae':
        model = gcae.GCAE(model_config)
    elif model_name == 'arcnn':
        model = arcnn.ARCNN(model_config)
    else:
        return None
    model.compile(word_embedding_matrix, aspect_embedding_matrix)
    logging.info('===={}模型加载完毕===='.format(model_name))
    return model

def train(train_x, train_y, valid_x, valid_y, embedding_matrix, model_list, label_name='reason',
          file_path='trained_models'):
    avg_acc = 0
    for model_name in model_list:
        config = load_config(model_name)
        config['num_classes'] = train_y.shape[1]
        config['max_length'] = train_x[0].shape[1]
        ytc = load_model(model_name, config, embedding_matrix[0], embedding_matrix[1])
        valid_pred, best_acc = ytc.fit(train_x, train_y, valid_x, valid_y, predicted=True,
                                       filename='{}/{}_{}.model'.format(file_path, label_name, model_name))
        avg_acc += best_acc
        logging.info('\t标签{}\t模型{}得分:\tacc:{}'.format(label_name,model_name, best_acc))
        del ytc
        if K.backend() == 'tensorflow':
            K.clear_session()
            rand_set()
    logging.info('\t标签{}平均分:\tacc:{}'.format(label_name, avg_acc / len(model_list)))


def predict(test_x, embedding_matrix, label_name, model_list, num_classes, use_ensemble=False, file_path='trained_models'):
    """
    载入预先训练模型，测试集预测。分为ensemble和single模式
    """
    if not use_ensemble:
        model_name = model_list[0]
        model_filename = '{}/{}_{}.model'.format(file_path, label_name, model_name)
        logging.info('\t开始预测模型{}'.format(model_name))
        config = load_config(model_name)
        config['num_classes'] = num_classes
        config['max_length'] = test_x[0].shape[1]
        ytc = load_model(model_name, config, embedding_matrix)
        ytc.model.load_weights(model_filename)
        y_pred = ytc.predict(test_x)
        return y_pred
    else:
        all_pred = []
        for model_name in model_list:
            model_filename = '{}/{}_{}.model'.format(file_path, label_name, model_name)
            logging.info('\t开始预测模型{}'.format(model_name))
            config = load_config(model_name)
            config['num_classes'] = num_classes
            config['max_length'] = test_x[0].shape[1]
            ytc = load_model(model_name, config, embedding_matrix[0], embedding_matrix[1])
            ytc.model.load_weights(model_filename)
            all_pred.append(ytc.predict(test_x))
            del ytc
        return all_pred


def load_test_x():
    id2subject = {0: '价格',
                  1: '操控',
                  2: '配置',
                  3: '安全性',
                  4: '油耗',
                  5: '动力',
                  6: '空间',
                  7: '外观',
                  8: '内饰',
                  9: '舒适性'}

    subject2id = {j: i for i, j in id2subject.items()}
    with open('data/subject_df.pkl', 'rb')as f:
        subject_train_df, subject_test_df, sentiment_train_df = pickle.load(f)
    sentiment_test_df = pd.read_csv('subject_test.csv')
    sentiment_test_df = pd.merge(sentiment_test_df, subject_test_df.iloc[:, :], on='content_id')
    aspect_x = np.array([subject2id[i] for i in sentiment_test_df.subject.tolist()]).reshape((-1, 1))
    with open('data/word_tokenizer.pkl', 'rb')as f:
        word_tokenizer = pickle.load(f)
    sentiment_word_sequence = word_tokenizer.texts_to_sequences(sentiment_test_df.words.tolist())
    test_x = pad_sequences(sentiment_word_sequence, maxlen=70, truncating='post')
    return [test_x, aspect_x]


def main(argv):
    """
    服务器后台运行: nohup python main.py --mode train --gpu 2  --label slot > out.log 2>&1 &
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='ensemble,train,test,search_para')
    parser.add_argument('--gpu', default='no', help='before running watch nvidia-smi')
    parser.add_argument('--embed', default='domain', help='domain,open,merge')
    parser.add_argument('--cut', default='word', help='char')
    parser.add_argument('--loggingmode', default='info', help='info:only result, debug:training details')
    args = parser.parse_args()
    # cut = args.cut
    emb = args.embed
    label_name = 'sentiment'

    if args.loggingmode == 'info':
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s%(message)s',
                            filename='trained_models/{}_{}.log'.format(args.mode, label_name),
                            filemode='a')
    else:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s%(message)s',
                            filename='trained_models/{}_{}.log'.format(args.mode, label_name),
                            filemode='a')
    if len(args.gpu) == 1:
        init_env(str(args.gpu))
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    rand_set()

    for cut in ['word']:
        train_x, train_y, valid_x, valid_y, domain_emb, open_emb, merge_emb, domain_aspect, open_aspect, \
                merge_aspect = load_data('data/nn_{}_{}.pkl'.format(label_name,cut))
        embedding_dict = {
            'domain': domain_emb,
            'open': open_emb,
            'merge': merge_emb,
        }

        aspect_dict = {
            'domain': domain_aspect,
            'open': open_aspect,
            'merge': merge_aspect,
        }

        word_emb = embedding_dict.get(emb)
        aspect_emb = aspect_dict.get(emb)
        embedding_matrix = [word_emb, aspect_emb]

        model_list = ['arcnn']
        trained_models_path = 'trained_models/{}_{}_{}'.format(label_name, cut, emb)
        if not os.path.exists(trained_models_path):
            os.mkdir(trained_models_path)
        if args.mode == 'train':
            train(train_x, train_y, valid_x, valid_y, embedding_matrix, model_list,
                  label_name=label_name, file_path=trained_models_path)
        elif args.mode == 'valid':
            predicted_label = predict(valid_x, embedding_matrix, label_name, model_list, train_y.shape[1],
                                      use_ensemble=True, file_path=trained_models_path)
            with open('result/valid_pred_list_{}_{}.pkl'.format(label_name, cut),'wb') as f:
                pickle.dump(predicted_label,f)

        elif args.mode == 'test':
            test_x = load_test_x()
            predicted_label = predict(test_x, embedding_matrix, label_name, model_list, train_y.shape[1],
                                      use_ensemble=True, file_path=trained_models_path)
            with open('result/test_pred_list_{}_{}.pkl'.format(label_name, cut),'wb') as f:
                pickle.dump(predicted_label,f)


if __name__ == '__main__':
    main(sys.argv)