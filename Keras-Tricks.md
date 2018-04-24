# Keras Tricks

####1. 关于训练集，验证集和测试集

- 验证集是从训练集中抽取出来用于调参的，在validation_split中设置
  - 用 Keras 的 `validation_split` 之前要記得把資料先弄亂，因為它會從資料的最尾端開始取，如果沒有弄亂的話切出來的資料 bias 會很大。可以使用 `np.shuffle` 來弄亂
- 测试集是和训练集无交集的，用于测试所选参数用于该模型的效果的。在evaluate函数里设置

####2. 查看模型的评价指标

```python
history_dict = history.history
history_dict.keys()
dict_keys(['val_acc', 'acc', 'val_loss', 'loss’])
```

####3. 保存keras输出的loss，val

```python
hist=model.fit(train_set_x,train_set_y,batch_size=256,shuffle=True,nb_epoch=nb_epoch,validation_split=0.1)
with open('log_sgd_big_32.txt','w') as f:
    f.write(str(hist.history))
```

####4. 绘制精度和损失曲线

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

# 或者
history.loss_plot('epoch')
```

####5. 将整型 label 转换成 one-hot 形式

```python
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
```

####6. 自制回调函数 callback

```python
# 该回调函数将在每个epoch后保存概率文件
from keras.callbacks import Callback
class WritePRF(Callback):
   def __init__(self, data):
      super(WritePRF, self).__init__()
      self.data = data
   def on_epoch_end(self, epoch, logs=None):
      
# 该回调函数将在每个迭代后保存的最好模型
from keras.callbacks import ModelCheckpoint  

checkpoint = ModelCheckpoint(  
    'model.h5',  
    monitor = 'val_loss',  
    verbose = 1,  
    save_best_only = True,  
    mode = 'min',  
) 
```

####7. Grid Search Hyperparameters 网格搜索

[Keras/Python深度学习中的网格搜索超参数调优（附源码）](http://geek.csdn.net/news/detail/95494)

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

####8. 编写自己的层

对于简单的定制操作，我们或许可以通过使用layers.core.Lambda层来完成。要定制自己的层，你需要实现下面三个方法:

- build(input_shape)：这是定义权重的方法
- call(x)：这是定义层功能的方法，除非你希望你写的层支持masking，否则你只需要关心call的第一个参数：输入张量
- compute_output_shape(input_shape)：如果你的层修改了输入数据的shape，你应该在这里指定shape变化的方法，这个函数使得Keras可以做自动shape推断

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

####9. keras保存和加载自定义损失模型

如果使用了自定义的loss函数， 则需要在加载模型的时候，指定load_model函数提供的一个custom_objects参数：在custom_objects参数词典里加入keras的未知参数，如：

```python
custom_objects={'ChainCRF': ClassWrapper, 'loss': loss, 'sparse_loss': sparse_loss}
model = load_model('model/tmpModel.h5', custom_objects=create_custom_objects())
```

####10. PRF 值计算

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

####11. keras 获取中间层的输出

```python
# 加载权重到当前模型
model = load_model(model_path)

'''获取中间层的输出'''
layer_name = 'my_layer'
intermediate_layer_model = Model(input=model.input,
                         output=model.get_layer(layer_name).output)

intermediate_output = intermediate_layer_model.predict(X_test)
print(type(intermediate_output))

with open('intermediate_output.txt', 'w') as f:
   for i in intermediate_output:
      f.write(i)
```

####12. 关于优化方法使用的问题

- 定义

```
"Optimization" refers to the process of adjusting a model to get the best performance possible on the training data (the "learning" in "machine learning"), 
```

- 优化方法使用
  - Adam，Adade，RMSprop结果都差不多，Nadam因为是adam的动量添加的版本，在收敛效果上会更出色。
  - 优化器公用参数 clipnorm 和 clipvalue
    - 参数一：clipnorm 对梯度进行裁剪，最大值为1
    - 参数二：clipvalue 对梯度范围进行裁剪，范围（-x，x）
  - 一般的起手式: Adam
  - Keras 推薦 RNN 使用 RMSProp
    - 在訓練 RNN 需要注意 explosive gradient 的問題 => clip gradient 的暴力美學
    - `opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)`
  - 可以直接在sgd声明函数中修改参数来直接修改学习率，学习率变化如下图：

```
sgd = SGD(lr=learning_rate, decay=learning_rate/nb_epoch, momentum=0.9, nesterov=True)
```

####13. 关于过拟合问题的讨论

- 首先是定义：

```
the performance (validation loss?) of our model on the held-out validation data would always peak after a few epochs and would then start degrading, i.e. our model would quickly start to _overfit_ to the training data.
```

- 防止过拟合的方法
  - 第一种就是添加dropout层，dropout可以放在很多类层的后面，用来抑制过拟合现象，常见的可以直接放在Dense层后面，一般在Dropout设置0.5。Dropout相当于Ensemble，dropout过大相当于多个模型的结合，一些差模型会拉低训练集的精度。
    - 通常只加在 hidden layer，不會加在 output layer，因為影響太大了，除非 output layer 的 dimension 很大。
    - Dropout 會讓 training performance 變差
    - 參數少時，regularization
  - 第二种是使用参数正则化，也就是在一些层的声明中加入L1或L2正则化系数，在一定程度上提升了模型的泛化能力。`kernel_regularizer=regularizers.l2(0.001)`
  - Reducing the network's size： The simplest way to prevent overfitting is to reduce the size of the model, i.e. the number of learnable parameters in the model
  - Early Stopping
    - 希望在 Model overfitting 之前就停止 training
    - Early Stopping in Keras
      - `from keras.callbacks import EarlyStopping`
      - `early_stopping=EarlyStopping(monitor='val_loss', patience=3)`

####14. Batchnormalization层的放置问题

BN层针对数据分布进行优化，对于BN来说其不但可以防止过拟合，还可以防止梯度消失等问题，并且可以加快模型的收敛速度，但是加了BN，模型训练往往会变得慢些。具体放置位置试！

```python
BatchNormalization(mode=0, axis=1)	# 输入是形如（samples，channels，rows，cols）的4D图像张量，需要设置axis=1
Dense()
BatchNormalization(mode=1)	# 按样本规范化，该模式默认输入为2D
```

####15. 关于泛化

while "generalization" refers to how well the trained model would perform on data it has never seen before. The goal of the game is to get good generalization, of course, but you do not control generalization; you can only adjust the model based on its training data.

####16. categorical_crossentropy损失函数 vs. sparse_categorical_crossentropy损失函数

- There are two ways to handle labels in multi-class classification: Encoding the labels via "categorical encoding" (also known as "one-hot encoding") and using `categorical_crossentropy` as your loss function. Encoding the labels as integers and using the `sparse_categorical_crossentropy` loss function.


####17. 通过生成器的方式训练模型，节省内存

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

####18. 多类别预测概率转换

`print np.argmax(y_pred, axis=-1)`

####19. CNN+LSTM的思考


- Because RNNs are extremely expensive for processing very long sequences, but 1D convnets are cheap, it can be a good idea to use a 1D convnet as a preprocessing step before a RNN, shortening the sequence and extracting useful representations for the RNN to process.


#### 20. 使用预训练模型的权重

```python
WEIGHTS_PATH = 'bottleneck_fc_model.h5'
model1.save_weights(WEIGHTS_PATH)
model2.load_weights(WEIGHTS_PATH)
# layer.trainable = False
model2.fit()
```



---

常见问题和解决方法

- ####如果训练中发现loss的值为NAN，这时可能的原因如下：

  - 学习率太高: loss爆炸, 或者nan
  - 学习率太小: 半天loss没反映
  - relu作为激活函数?
  - 如果是自己定义的损失函数，这时候可能是你设计的损失函数有问题
  - [训练DL，出现Loss=nan](evernote:///view/1937456/s4/177cd2ff-02a7-4294-8081-8454a219fcfc/177cd2ff-02a7-4294-8081-8454a219fcfc/)

- ####loss为负数 

  - 如果出现loss为负，是因为之前多分类的标签哪些设置不对，现在是5分类的，写成了2分类之后导致了Loss为负数
  - 也可能是损失函数选择错误导致

- ####ResourceExhaustedError: OOM when allocating tensor with shape

意思就是GPU的内存不够了，检查下是否有其他程序占用，不行就重启下IDE，或kill 进程ID

- [Failing to Implement a Custom Objective Function #4920](https://github.com/fchollet/keras/issues/4920)

- ####keras指定显卡且限制显存用量

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

  需要注意的是，虽然代码或配置层面设置了对显存占用百分比阈值，但在实际运行中如果达到了这个阈值，程序有需要的话还是会突破这个阈值。换而言之如果跑在一个大数据集上还是会用到更多的显存。以上的显存限制仅仅为了在跑小数据集时避免对显存的浪费而已。


- ####Keras 切换后端（Theano和TensorFlow）

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

