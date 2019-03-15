# AspectSentimentZoo
#### 项目介绍
面向评价对象的情感分类模型，Keras-TensorFlow

#### 软件架构
```
|-- sentiment_models
    |-- atae.py  # atae
|-- main.py           # 模型训练和测试
```


#### Referenced Paper
1.[Aspect Based Sentiment Analysis with Gated Convolutional Networks - ACL2018](sentiment_models/gcae.py)
2.[Attention-based LSTM for Aspect-level Sentiment Classification - EMNLP 2016](sentiment_models/atae.py)
3.[Aspect Level Sentiment Classification with Deep Memory Network - ACL 2016](sentiment_models.memnn.py)\
4.[Connecting Targets to Tweets: Semantic Attention-based Model for Target-specific Stance Detection](sentiment_models.arcnn.py)
5.[Stance Classification with Target Specific Neural Attention Networks -IJCAI2017](sentiment_models/tan.py)

#### 使用说明
2. python main.py --mode train  训练NN模型
3. python main.py --mode test   测试集预测输出

#### TODO
- [ ] 整理代码

