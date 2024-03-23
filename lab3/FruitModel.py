import math
from SST_2.dataset import traindataset, minitraindataset
from fruit import get_document, tokenize
import pickle
import numpy as np
from importlib.machinery import SourcelessFileLoader
from autograd.BaseGraph import Graph
from autograd.BaseNode import *
import sys

class NullModel:
    def __init__(self):
        pass

    def __call__(self, text):
        return 0


class NaiveBayesModel:
    def __init__(self):
        self.dataset = traindataset() # 完整训练集
        # self.dataset = minitraindataset() # 用来调试的小训练集

        # 以下内容可根据需要自行修改
        self.token_num = [{}, {}] # token在正负样本中出现次数
        self.V = 0 #语料库token数量
        self.pos_neg_num = [0, 0] # 正负样本数量
        self.count()

    def count(self):
        # TODO: YOUR CODE HERE
        # 提示：统计token分布不需要返回值
        for text, label in self.dataset:
            # print(text)
            # print(label)
            label=int(label)
            self.pos_neg_num[label]+=1
            for word in text:
                if word in self.token_num[label]:
                    self.token_num[label][word]+=1
                else:
                    self.token_num[label][word]=1
                    if word not in self.token_num[label]:
                        self.V+=1
        # TODO: YOUR CODE HERE

    def __call__(self, text):
        # TODO: YOUR CODE HERE
        # 返回1或0代表当前句子分类为正/负样本
        neg_val=self.pos_neg_num[0]
        pos_val=self.pos_neg_num[1]
        for word in text:
            neg_val*=(self.token_num[0].get(word,0)+1)/(self.pos_neg_num[0]+self.V)
            pos_val*=(self.token_num[1].get(word,0)+1)/(self.pos_neg_num[1]+self.V)
        return 1 if pos_val>neg_val else 0
        # TODO: YOUR CODE HERE


def buildGraph(dim, num_classes): #dim: 输入一维向量长度， num_classes:分类数
    # TODO: YOUR CODE HERE
    # 填写网络结构，请参考lab2相关部分
    # 填写代码前，请确定已经将lab2写好的BaseNode.py与BaseGraph.py放在autograd文件夹下 
    
    nodes = [
             Linear(dim, dim-20),
             Dropout(),
            #  relu(),
            #  sigmoid(),
             
             Linear(dim-20, dim-30),
             Dropout(),
             
             Linear(dim-30, dim-40),
             Dropout(),
            #  relu(),

             Linear(dim-40, dim-60),
             Dropout(),
            #  relu(),

            #  Linear(dim-60, dim-80),
            #  Dropout(),
            #  relu(),
            #  sigmoid(),
            #  Linear(dim-40, dim-50),
            #  Dropout(),
             
             Linear(dim-60, num_classes),
             LogSoftmax(),
             NLLLoss(np.zeros(10))]
    graph = Graph(nodes)
    return graph
    # TODO: YOUR CODE HERE


save_path = "model/mlp.npy"

class Embedding():
    def __init__(self):
        self.emb = dict() 
        with open("words.txt", encoding='utf-8') as f: #word.txt存储了每个token对应的feature向量，self.emb是一个存储了token-feature键值对的Dict()，可直接调用使用
            for i in range(50000):
                row = next(f).split()
                word = row[0]
                vector = np.array([float(x) for x in row[1:]])
                self.emb[word] = vector
        
    def __call__(self, text):
        # TODO: YOUR CODE HERE
        # 利用self.emb将句子映射为一个一维向量，注意，同时需要修改训练代码中的网络维度部分
        # 数据格式可以参考lab2
        ans=np.zeros(100)
        cnt=1
        for word in text:
            if word in self.emb:
                ans=np.add(ans,self.emb[word])
                cnt+=1
        return ans/cnt
        # TODO: YOUR CODE HERE


class MLPModel():
    def __init__(self):
        self.embedding = Embedding()
        with open(save_path, "rb") as f:
            self.network = pickle.load(f)
        self.network.eval()
        self.network.flush()

    def __call__(self, text):
        X = self.embedding(text)
        pred = self.network.forward(X, removelossnode=1)[-1]
        haty = np.argmax(pred, axis=0)
        return haty


class QAModel():
    def __init__(self):
        self.document_list = get_document()

    def tf(self, word, document):# word:单词，document:文档，是一个list
        # TODO: YOUR CODE HERE
        # 返回单词在文档中的频度
        return math.log10(document.count(word)/len(document)+1)
        raise NotImplementedError  

    def idf(self, word):
        # TODO: YOUR CODE HERE
        # 返回单词IDF值，提示：你需要利用self.document_list来遍历所有文档
        # a = sum([1 for document in self.document_list if word in document])
        # assert a == 0, a
        # return math.log10(len(self.document_list)/(a+1))
        return math.log10(len(self.document_list)/(sum([1 for document in self.document_list if word in document['document']])+1))
        
        raise NotImplementedError  
    
    def tfidf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回TF-IDF值
        # print(self.idf(word))
        return self.tf(word,document)*self.idf(word)
        raise NotImplementedError  

    def __call__(self, query):
        query = tokenize(query) # 将问题token化
        # TODO: YOUR CODE HERE
        # 利用上述函数来实现QA
        # 提示：你需要根据TF-IDF值来选择一个最合适的文档，再根据TF-IDF值选择最合适的句子
        # 返回时请返回原本句子，而不是token化后的句子，可以参考README中数据结构部分以及fruit.py中用于数据处理的get_document()函数
        score_document=np.zeros(len(self.document_list))#每个文档的得分列表
        for i in range(len(self.document_list)):#计算每个文档的得分
            for word in query:
                score_document[i]+=self.tfidf(word,self.document_list[i]['document'])
        opt_doc_index=np.argmax(score_document)#得分最高的文档的索引
        
        sentence_list=self.document_list[opt_doc_index]['sentences']#得分最高的文档的句子列表
        score_sentence=np.zeros(len(sentence_list))#得分最高的文档的句子得分列表
        for i in range(len(sentence_list)):#计算得分最高的文档的每个句子的得分
            for word in query:
                tf0=math.log10(sentence_list[i][0].count(word)/len(sentence_list)+1)
                idf0=math.log10(len(sentence_list)/(sum([1 for sentence in sentence_list if word in sentence[0]])+1))
                # score_sentence[i]+=word in sentence_list[i][0]  
                score_sentence[i]+=tf0*idf0
        opt_sentence_index=np.argmax(score_sentence)#得分最高的文档的得分最高的句子的索引
        
        ans=sentence_list[opt_sentence_index][1]#得分最高的文档的得分最高的句子
        return ans
        raise NotImplementedError
            

modeldict = {
    "Null": NullModel,
    "Naive": NaiveBayesModel,
    "MLP": MLPModel,
    "QA": QAModel,
}


if __name__ == '__main__':
    embedding = Embedding()
    lr = 6e-4   # 学习率
    wd1 = 1e-4  # L1正则化
    wd2 = 0  # L2正则化
    batchsize = 64
    max_epoch = 35
    dp = 0.1

    graph = buildGraph(100, 2) # 维度需要自己修改

    # 训练
    best_train_acc = 0
    # dataloader = traindataset() # 完整训练集
    dataloader = minitraindataset() # 用来调试的小训练集
    for i in range(1, max_epoch+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        X = []
        Y = []
        cnt = 0
        for text, label in dataloader:
            x = embedding(text)
            label = np.zeros((1)).astype(np.int32) + label
            X.append(x)
            Y.append(label)
            cnt += 1
            if cnt == batchsize:
                X = np.concatenate(X, 0)
                Y = np.concatenate(Y, 0)
                X = X.reshape(-1, 100)
                graph[-1].y = Y
                graph.flush()
                pred, loss = graph.forward(X)[-2:]
                hatys.append(np.argmax(pred, axis=1))
                ys.append(Y)
                graph.backward()
                graph.optimstep(lr, wd1, wd2)
                losss.append(loss)
                cnt = 0
                X = []
                Y = []

        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)