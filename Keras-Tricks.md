# Keras Tricks

- 查看模型的评价指标

```python
history_dict = history.history
history_dict.keys()
dict_keys(['val_acc', 'acc', 'val_loss', 'loss’])
```



- 将整型 label 转换成 one-hot 形式

```python
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
```



- There are two ways to handle labels in multi-class classification: Encoding the labels via "categorical encoding" (also known as "one-hot encoding") and using categorical_crossentropy as your loss function. Encoding the labels as integers and using the sparse_categorical_crossentropy loss function.
- Overfitting: the performance (validation loss?) of our model on the held-out validation data would always peak after a few epochs and would then start degrading, i.e. our model would quickly start to _overfit_ to the training data.
- "Optimization" refers to the process of adjusting a model to get the best performance possible on the training data (the "learning" in "machine learning"), 
- while "generalization" refers to how well the trained model would perform on data it has never seen before. The goal of the game is to get good generalization, of course, but you do not control generalization; you can only adjust the model based on its training data.
- The processing of fighting overfitting in this way is called regularization.



- 防止过拟合的方法：

  - Reducing the network's size： The simplest way to prevent overfitting is to reduce the size of the model, i.e. the number of learnable parameters in the model

  - Adding weight regularization

    L1 regularization

    L2 regularization：`kernel_regularizer=regularizers.l2(0.001)`

  - Adding dropout
    dropout是从模型结构方面优化
    常见的可以直接放在--Dense层后面，Convolutional和Maxpooling之间

    The "dropout rate" is usually set between 0.2 and 0.5.

  - Batchnormalization
    BN针对数据分布进行优化

    对于BN来说其不但可以防止过拟合，还可以防止梯度消失等问题，并且可以加快模型的收敛速度，但是加了BN，模型训练往往会变得慢些。



- Because RNNs are extremely expensive for processing very long sequences, but 1D convnets are cheap, it can be a good idea to use a 1D convnet as a preprocessing step before a RNN, shortening the sequence and extracting useful representations for the RNN to process.



- 自制 回调函数 callback
  输出训练过程中训练集合验证集准确值和损失值得变化

```python
# 该回调函数将在每个epoch后保存概率文件
from keras.callbacks import Callback
class WritePRF(Callback):
   def __init__(self, filepath, data, label):
      super(WritePRF, self).__init__()
      self.filepath = filepath
      self.data = data
      self.label = label

   def on_epoch_end(self, epoch, logs=None):
      resultFile = self.filepath + str(epoch) + '.txt'
      predictions = self.model.predict(self.data)
      PRF.calculate(predictions, self.label, resultFile)
      
# 该回调函数将在每个迭代后保存的最好模型
class checkpoint():
  def __init__(self, model_file):
      self.model_file = model_file

  def check(self):
      checkpoint = ModelCheckpoint(filepath=self.model_file, monitor='val_loss',
                  verbose=1, save_best_only=True, mode='min')
      return checkpoint

write_call = WritePRF(filepath=RESULT_FILE, data=X_test, label=test_label)
```



- 绘制精度和损失曲线

```python
import matplotlib.pyplot as plt

def plot(history):
   plt.figure(figsize=(16,7))
   plt.subplot(121)
   plt.xlabel('epoch')
   plt.ylabel('acc')
   plt.plot(history.epoch, history.history['acc'], 'b', label="acc")
   plt.plot(history.epoch, history.history['val_acc'], 'r', label="val_acc")
   plt.scatter(history.epoch, history.history['acc'], marker='*')
   plt.scatter(history.epoch, history.history['val_acc'])
   plt.legend(loc='lower right')

   plt.subplot(122)
   plt.xlabel('epoch')
   plt.ylabel('loss')
   plt.plot(history.epoch, history.history['loss'], 'b', label="loss")
   plt.plot(history.epoch, history.history['val_loss'], 'r', label="val_loss")
   plt.scatter(history.epoch, history.history['loss'], marker='*')
   plt.scatter(history.epoch, history.history['val_loss'], marker='*')
   plt.legend(loc='lower right')
   plt.show()
```

或者，

`history.loss_plot('epoch')`



- [Keras/Python深度学习中的网格搜索超参数调优（附源码）](http://geek.csdn.net/news/detail/95494)

- Grid Search Hyperparameters 网格搜索

```python
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model
 
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
# define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# define the grid search parameters
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
neurons = [1, 5, 10, 15, 20, 25, 30]

param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

```



- Aetros的插件，[Aetros网址](http://aetros.com/)，这是一个基于Keras的一个管理工具，可以可视化你的网络结构，中间卷积结果的可视化，以及保存你以往跑的所有结果



- 训练集需要一开始shuffle ；
  验证集的划分只要在fit函数里设置validation_split的值就好了
  测试集的使用只要在evaluate函数里设置就好了



- K层交叉检验（k-fold Cross Validation）



- 如果训练中发现loss的值为NAN，这时可能的原因如下：
  - 学习率太高: loss爆炸, 或者nan
  - 学习率太小: 半天loss没反映
  - relu作为激活函数?
  - 如果是自己定义的损失函数，这时候可能是你设计的损失函数有问题
  - [训练DL，出现Loss=nan](evernote:///view/1937456/s4/177cd2ff-02a7-4294-8081-8454a219fcfc/177cd2ff-02a7-4294-8081-8454a219fcfc/)



- keras指定显卡且限制显存用量

  keras在使用GPU的时候有个特点，就是默认全部占满显存。需要修改后端代码：

  - 方法1：使用固定显存的GPU

  ```python
  import tensorflow as tf
  from keras.backend.tensorflow_backend import set_session

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.3
  config.gpu_options.allow_growth=True
  set_session(tf.Session(config=config))
  ```

  ​

  需要注意的是，虽然代码或配置层面设置了对显存占用百分比阈值，但在实际运行中如果达到了这个阈值，程序有需要的话还是会突破这个阈值。换而言之如果跑在一个大数据集上还是会用到更多的显存。以上的显存限制仅仅为了在跑小数据集时避免对显存的浪费而已。

  ​


- Keras 切换后端（Theano和TensorFlow）

  ~/.keras/keras.json

```python
用thesorflow的话，keras.json写入
{
    "image_dim_ordering": "tf", 
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "backend": "tensorflow"
}
```





- 优化器公用参数 clipnorm 和 clipvalue

  \#参数一：clipnorm 对梯度进行裁剪，最大值为1

  \#参数二：clipvalue 对梯度范围进行裁剪，范围（-x，x）



- 保存自定义模型

  如果使用了自定义的loss函数， 则需要在加载模型的时候，指定load_model函数提供的一个custom_objects参数：在custom_objects参数词典里加入keras的未知参数，如：

  ```python
  custom_objects={'ChainCRF': ClassWrapper, 'loss': loss, 'sparse_loss': sparse_loss}
  model = load_model('model/tmpModel.h5', custom_objects=create_custom_objects())
  ```

  ​

- 通过生成器的方式训练模型，节省内存

```python
#从节省内存的角度，通过生成器的方式来训练
    def data_generator(data, chars, targets, data_a, chars_a, targets_a, batch_size): 
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)//batch_size)]
        while True:
            for i in batches:
                xx, yy = np.array(data[i]), np.array(targets[i])
                char = np.array(chars[i])
                xx_a, yy_a = np.array(data_a[i]), np.array(targets_a[i])
                char_a = np.array(chars_a[i])
                yield ([xx, char, xx_a, char_a], [yy, yy_a])

    print('Build model...')
    model = buildModel(max_word)

    generator = data_generator(train_x, train_char, train_y, aux_train_x, aux_train_char, aux_train_y, batch_size)
    samples_per_epoch = len(train_x)
    steps_per_epoch = samples_per_epoch // batch_size
    # StopIteration: dataset fully readed before fit end
    history = model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs)
```



- 编写自己的层
  - build(input_shape)：这是定义权重的方法
  - call(x)：这是定义层功能的方法
  - compute_output_shape(input_shape)：如果你的层修改了输入数据的shape，你应该在这里指定shape变化的方法

```python
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```



- PRF 值计算

```python
from __future__ import print_function
import numpy as np

def calculate(predictions, test_label, RESULT_FILE):
   num = len(predictions)
   with open(RESULT_FILE, 'w') as f:
      for i in range(num):
         if predictions[i][1] > predictions[i][0]:
            predict = +1
         else:
            predict = -1
         f.write(str(predictions[i][0]) + ' ' + str(predictions[i][1]) + '\n')
      # f.write(str(predict) + str(predictions[i]) + '\n')

   TP = len([1 for i in range(num) if
           predictions[i][1] > predictions[i][0] and (test_label[i] == np.asarray([0, 1])).all()])
   FP = len([1 for i in range(num) if
           predictions[i][1] > predictions[i][0] and (test_label[i] == np.asarray([1, 0])).all()])
   FN = len([1 for i in range(num) if
           predictions[i][1] < predictions[i][0] and (test_label[i] == np.asarray([0, 1])).all()])
   TN = len([1 for i in range(num) if
           predictions[i][1] < predictions[i][0] and (test_label[i] == np.asarray([1, 0])).all()])

   precision = recall = Fscore = 0, 0, 0
   try:
      precision = TP / (float)(TP + FP)  # ZeroDivisionError: float division by zero
      recall = TP / (float)(TP + FN)
      Fscore = (2 * precision * recall) / (precision + recall)
   except ZeroDivisionError as exc:
      print(exc.message)

   print(">> Report the result ...")
   print("-1 --> ", len([1 for i in range(num) if predictions[i][1] < predictions[i][0]]))
   print("+1 --> ", len([1 for i in range(num) if predictions[i][1] > predictions[i][0]]))
   print("TP=", TP, "  FP=", FP, " FN=", FN, " TN=", TN)
   print('\n')
   print("precision= ", precision)
   print("recall= ", recall)
   print("Fscore= ", Fscore)
```

