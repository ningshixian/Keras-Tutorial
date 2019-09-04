'''
Implement fit_generator( ) in Keras

https://github.com/keras-team/keras/issues/107

# let's say you have a BatchGenerator that yields a large batch of samples at a time
# (but still small enough for the GPU memory)
for e in range(nb_epoch):
    print("epoch %d" % e)
    for X_train, Y_train in BatchGenerator(): 
        model.fit(X_batch, Y_batch, batch_size=32, nb_epoch=1)


# Alternatively, let's say you have a MiniBatchGenerator that yields 32-64 samples at a time:
for e in range(nb_epoch):
    print("epoch %d" % e)
    for X_train, Y_train in MiniBatchGenerator(): # these are chunks of ~10k pictures
        model.train(X_batch, Y_batch)


http://hironsan.hatenablog.com/entry/2017/09/09/130608
'''


import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb


def batch_iter(data, labels, batch_size, shuffle=True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        data_size = len(data)
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]   # np.array
                shuffled_labels = labels[shuffle_indices]   # np.array
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X, y = shuffled_data[start_index: end_index], shuffled_labels[start_index: end_index]
                yield X, y

    return num_batches_per_epoch, data_generator()


def main(mode):
    max_features = 20000
    maxlen = 80
    batch_size = 32

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    if mode == 'fit':
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test))
    else:
        train_steps, train_batches = batch_iter(x_train, y_train, batch_size)
        valid_steps, valid_batches = batch_iter(x_test, y_test, batch_size)
        model.fit_generator(train_batches, train_steps, epochs=1, validation_data=valid_batches, validation_steps=valid_steps)

    # model.predict_generator(train_batches, train_steps)


if __name__ == '__main__':
    main('fit')