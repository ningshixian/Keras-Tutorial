# Keras 第三方资源+项目

- [Elephas](https://github.com/maxpumperla/elephas): Distributed Deep Learning with Keras & Spark

- [Hyperas](https://github.com/maxpumperla/hyperas): Hyperparameter optimization

- [Hera](https://github.com/jakebian/hera): in-browser metrics dashboard for Keras models

- [Kerlym](https://github.com/osh/kerlym): reinforcement learning with Keras and OpenAI Gym

- [Qlearning4K](https://github.com/farizrahman4u/qlearning4k): reinforcement learning add-on for Keras

- [seq2seq](https://github.com/farizrahman4u/seq2seq): Sequence to Sequence Learning with Keras

- [Seya](https://github.com/EderSantana/seya): Keras extras

- [Keras Language Modeling](https://github.com/codekansas/keras-language-modeling): Language modeling tools for Keras

- [Recurrent Shop](https://github.com/datalogai/recurrentshop): Framework for building complex recurrent neural networks with Keras

- [Keras.js](https://github.com/transcranial/keras-js): Run trained Keras models in the browser, with GPU support

- [keras-vis](https://github.com/raghakot/keras-vis): Neural network visualization toolkit for keras.

- [Keras科研扩展集](http://t.cn/R9AXSK5)

- [Learning Deep Learning with Keras](http://t.cn/RXs9LXM)

  用Keras 学习深度学习：汇集了许多深度学习的资源。

- ​



###Projects built with Keras

- [NMT-Keras](https://github.com/lvapeab/nmt-keras): Neural Machine Translation using Keras.
- [snli-entailment](https://github.com/shyamupa/snli-entailment): Independent implementation of attention model for textual entailment from the paper ["Reasoning about Entailment with Neural Attention"](http://arxiv.org/abs/1509.06664).
- [Headline generator](https://github.com/udibr/headlines): independent implementation of [Generating News Headlines with Recurrent Neural Networks](http://arxiv.org/abs/1512.01712)
- [Conx](https://conx.readthedocs.io/) - easy-to-use layer on top of Keras, with visualizations (eg, no knowledge of numpy needed)
- [神经张量网络NTN-文本实体关系探究-Keras](http://deeplearn-ai.com/2017/11/21/neural-tensor-network-exploring-relations-among-text-entities/?i=2)
- [keras实现BiLSTM+CNN+CRF文字标记NER](http://blog.csdn.net/xinfeng2005/article/details/78485748)
- Linear Chain CRF layer 实现
  - [Linear Chain CRF layer and a text chunking example#4621](https://github.com/pressrelations/keras/blob/98b2bb152b8d472150a3fc4f91396ce7f767bed9/keras/layers/crf.py)
  - [CRF for keras 2.0 #6226](https://github.com/fchollet/keras/pull/6226/commits/8c10628875a8190a7eab596fcf524a7dff346366)
  - [CRF for keras 2.x #76](https://github.com/farizrahman4u/keras-contrib/pull/76/files/3256eec4c113cbc6709b7cce3bd0edaca7cd3730#diff-907aee041113980abb723685f90356fb)
- Dynamic k-Max Pooling #373

```python
from keras.engine import Layer, InputSpec
from keras.layers import Flatten
import tensorflow as tf

class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])
        
        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]
        
        # return flattened output
        return Flatten()(top_k)
```



