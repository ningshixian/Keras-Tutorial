'''
预处理
 1、从.eval和.in文件中获取数据和标签文件，初步过滤
 2、对句子中所有的实体做上标记（在实体前后加入特定标签[BEGIN-/-INSIDE]）
 3、利用GENIA tagger工具对标记过的语料进行预处理，得到 test.out 文件
 4、根据预处理语料的标记（#@）获取BIO标签，写入test.out 文件

 问题：
 暂时无法解决嵌套实体的BIO标记
'''
import re
import codecs
from collections import defaultdict


def clean(data):
    data = data.replace('-', ' - ')
    data = data.replace('--', ' -- ')
    data = data.replace('/', ' / ')
    data = re.sub(r'\d+(?:\.\d+)?(%)?', '1', data)
    return data


def readData(dataFile='segmentData/test.in'):
    id_list = []
    id2sen = defaultdict()
    with codecs.open(dataFile, 'r', 'utf-8') as f:
        for line in f:
            sen_id = line.split()[0]
            sen = ' '.join(line.split()[1:])
            sen = sen.strip('\n').strip('.').strip('?').strip('!')
            sen = clean(sen)
            id_list.append(sen_id)
            id2sen[sen_id] = sen
    return id2sen, id_list


# 读取标签文件中 实体对应的 下标和实体名
def readGeneFile(geneFile='segmentData/GENE.eval'):
    id2gene_sorted = {}
    id2gene = {}
    id2idx = {}

    with codecs.open(geneFile, 'r', 'utf-8') as f:
        for line in f:
            splited = line.split('|')  # .encode('utf-8')
            ID = str(splited[0])
            index = (splited[1].split()[0], splited[1].split()[1])
            gene = str(splited[-1].strip('\n'))
            gene = clean(gene)

            if ID not in id2idx:
                id2idx[ID] = []
            id2idx[ID].append(index)

            if ID not in id2gene:
                id2gene[ID] = []
            id2gene[ID].append(gene)

    for k, v in id2gene.items():
        # 逆序排序
        temp = sorted(v, key=lambda x: len(x.split()), reverse=True)
        id2gene_sorted[k] = temp

    return id2gene_sorted


# 对句子中所有的实体做上标记（在实体前后加入特定标签[BEGIN-/-INSIDE]）
def replace(id2sen, id2gene):
    ng = 0
    sentences_flag = {}  # 替换后的句子
    for ID, sentence in id2sen.items():
        temp = []
        if ID in id2gene:
            for entity in id2gene[ID]:
                if entity not in temp:
                    temp.append(entity)
                    sentence = sentence.replace(entity, 'BEGIN-'+entity+'-INSIDE')
        sentences_flag[ID] = sentence

    with open('segmentData/test_flag.txt', 'w') as f:
        # 将原始语料 每一句前面的ID去掉 加入实体flag 写入文件
        for ID in id_list:
            f.write(sentences_flag[ID])
            f.write('\n')
    return sentences_flag


# 利用GENIA tagger工具对标记过的语料进行预处理（分词+POS+CHUNK+NER）
# 得到 test.out 文件


# 根据预处理语料的标记（#@）获取BIO标签
def getLabel(dataFile='test.genia'):
    flag = 0
    label = []
    label_sen = []
    sent = []
    with open(dataFile, 'r') as data:
        for line in data:
            if not line=='\n':
                word = line.split('\t')[0]
                if word.startswith('BEGIN-') and word.endswith('-INSIDE'):
                    if flag:
                        label_sen.append('I')
                    else:
                        label_sen.append('B')
                    flag = 0
                # elif word.endswith('-INSIDE'):
                elif '-INSIDE' in word:
                    label_sen.append('I')
                    flag=0
                elif word.startswith('BEGIN-'):
                    label_sen.append('B')
                    flag=1
                else:
                    if flag:
                        label_sen.append('I')
                    else:
                        label_sen.append('O')
                word = word.replace('BEGIN-', '').replace('-INSIDE', '')
                sent.append(word + '\t' + '\t'.join(line.split('\t')[1:-1]) + '\t' + label_sen[-1] + '\n')
            else:
                label.append(label_sen)
                label_sen = []
                sent.append('\n')
    with open('test.out', 'w') as f:
        # 将原始语料 每一句前面的ID去掉 加入实体flag 写入文件
        for line in sent:
            f.write(line)
    return label


if __name__ == '__main__':
    
    # id2sen, id_list = readData()
    # id2gene = readGeneFile()
    # print(id2sen['BC2GM022402970'])
    # data = replace(id2sen, id2gene)
    # print(data['BC2GM022402970'])

    label = getLabel()
    print(label[1114])

    

