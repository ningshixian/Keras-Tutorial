# Keras Attention 实现

### Attention 资料

如果你想进一步地学习如何在LSTM/RNN模型中加入attention机制，可阅读以下论文：

- [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)

- [Deep Language Modeling for Question Answering using Keras](https://codekansas.github.io/blog/2016/language.html)

  简单介绍了Keras的使用，以及细致讲解了简单的 Attentional LSTM 模型实现！！！

- [Attention and memory in deep learning and NLP](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)

- [Attention Mechanism](https://blog.heuritech.com/2016/01/20/attention-mechanism/)

- [Survey on Attention-based Models Applied in NLP](http://yanran.li/peppypapers/2015/10/07/survey-attention-model-1.html)

- [What is exactly the attention mechanism introduced to RNN?](https://www.quora.com/What-is-exactly-the-attention-mechanism-introduced-to-RNN-recurrent-neural-network-It-would-be-nice-if-you-could-make-it-easy-to-understand) （来自Quora）

- [What is Attention Mechanism in Neural Networks?](https://www.quora.com/What-is-Attention-Mechanism-in-Neural-Networks)



### Attention 实现

目前Keras官方还没有单独将attention模型的代码开源，下面有一些第三方的实现：

- [Deep Language Modeling for Question Answering using Keras](http://ben.bolte.cc/blog/2016/language.html)

- [Attention Model Available!](https://github.com/fchollet/keras/issues/2067)

- [Keras Attention Mechanism](https://github.com/philipperemy/keras-attention-mechanism)

  - attention_lstm
  - attention_dense

- [Keras Blstm+attention 的简单实现](http://blog.csdn.net/u010041824/article/details/78855435)

- [keras-language-modeling](https://github.com/codekansas/keras-language-modeling/blob/master/keras_models.py)

  代码的讲解对应上面的资料《Deep Language Modeling for Question Answering using Keras》

- [Attention and Augmented Recurrent Neural Networks](https://github.com/fchollet/keras/issues/1472)

- [How to add Attention on top of a Recurrent Layer (Text Classification)](https://github.com/fchollet/keras/issues/4962)

- [Attention Mechanism Implementation Issue](https://github.com/fchollet/keras/issues/1472)

- [Implementing simple neural attention model (for padded inputs)](https://github.com/fchollet/keras/issues/2612)

- [Attention layer requires another PR](https://github.com/fchollet/keras/issues/1094)

- [seq2seq library](https://github.com/farizrahman4u/seq2seq)


- [基于Attention Model的Aspect level文本情感分类---用Python+Keras实现](http://blog.csdn.net/orlandowww/article/details/53897634)
- [Keras 自注意力层实现](http://blog.csdn.net/mpk_no1/article/details/72862348)
- [可视化您的递归神经网络以及关注Keras](https://medium.com/datalogue/attention-in-keras-1892773a4f22) ” 的帖子提供了支持，GitHub项目名为“ [keras-attention](https://github.com/datalogue/keras-attention) ”