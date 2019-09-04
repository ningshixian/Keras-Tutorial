import json
import os
import numpy as np

from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        # 如果希望对文件中的内容进行预处理，均可以在 MySentences 迭代器中完成
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

# 输入 （迭代器的方式读取数据，节省 RAM 空间）

data_path = './EMBEDDINGS_TEXT_PATH/'
sentences = MySentences(data_path)  # a memory-friendly iterator

# 训练

model = Word2Vec(sentences,size=100,min_count=1,window=5,sg=1,iter=25,workers=20,negative=8)

# 预先归一化，使得词向量不受尺度影响

model.init_sims(replace=True) 

# 模型保存和加载

model.wv.save_word2vec_format('result/w2v.model.bin', binary=True)
model.wv.save_word2vec_format('result/w2v.model.txt', binary=False)
model1 = KeyedVectors.load_word2vec_format('result/w2v.model.txt', binary=False)
model2 = KeyedVectors.load_word2vec_format('result/w2v.model.bin', binary=True)
del model

# 载入模型后用更多的预料对其进行在线训练

# model.save('/tmp/mymodel')
# new_model = Word2Vec.load('/tmp/mymodel')
# model.train(more_sentences)

# 相似度计算
print(model1.similarity('of', 'a'))

# 查询词向量
print(model1['of'] )

# 词频统计
print(model1.vocab['of'])

# 词向量维度/字典大小
embeddingDim = model1.vector_size

# 词向量lookup矩阵
idx2vec = model1.vectors

# 词索引
word2idx = dict([(k, v.index) for k, v in model1.vocab.items()])
idx2word = dict([(v.index, k) for k, v in model1.vocab.items()])

# 词向量lookup字典
word2vec = dict([(idx2word[i], v) for i, v in enumerate(model1.vectors)])
word2vec['padding'] = np.zeros(embeddingDim)
word2vec['unknown'] = np.random.uniform(-0.1,0.1, size=embeddingDim)

# 获取训练语料的词索引字典vocab

# max_nb_word = 1000000
# texts = ['sentence1', 'sentence2', '...']

# from keras.preprocessing.text import Tokenizer
# tokenizer = Tokenizer(num_words=max_nb_word, filters=' ')
# tokenizer.fit_on_texts(texts)
# vocab = tokenizer.word_index

# keras词向量层权重矩阵

def get_embdding_weight():
    vocab = {'i':0, 'am':1, 'a':2, 'boy':3}
    vocabSize = len(vocab)

    embeddingWeights = np.zeros((vocabSize + 1, embeddingDim))
    for word, index in vocab.items():
        if word in word2vec:
            e = word2vec[word]
        else:
            e = word2vec['unknown']
        embeddingWeights[index, :] = e
    print(embeddingWeights)


