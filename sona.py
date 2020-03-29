import numpy as np
import operator
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import math
import matplotlib.lines as mlines
from csv import reader
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def str_column_to_int(dataset):
    for i in range(len(dataset)):
        if dataset[i][60]=='R':
            dataset[i][60]=1
        else:
            dataset[i][60]=2

filename = 'sonar.all-data.csv'
dataset = load_csv(filename)
for i in range(0, len(dataset[0])-1):
     str_column_to_float(dataset, i)
str_column_to_int(dataset)
def gety(dataset):
    y=[]
    for i in dataset:
        y.append(i[60])
    return y
y=gety(dataset)
y=np.array(y)
for i in dataset:
    i.pop()
dataset=np.array(dataset)
#dataset为数据
#y为标签
#print(len(dataset))
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#自己编写
#daice为待分类样本，dataset为已知样本，label为已知样本的标签，k为取前K个最近的样本
def classify0(daice,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    #print(dataSetSize)
    #下一条语句利用tile函数把待分类的样本扩充为矩阵，便于做减法
    diffMat = np.tile(daice,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    #对每个元素求平方
    sqDistances = sqDiffMat.sum(axis=1)
    #对每行求和
    distances = sqDistances ** 0.5
    #对平方距离进行开平方，求欧氏距离
    sortedDistIndicies = distances.argsort()
    #对距离从小到大进行排序，返回元素的下标
    classCount = {}
    #构造一个字典，对前k个距离最小的点统计他们所属的类别
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #print(voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
    #print(classCount)
    #
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    #print(sortedClassCount)
    return sortedClassCount[0][0]

#自己写
cis=0
KF=KFold(n_splits=20,shuffle=True)
fenshu=[]
for k in range(1,51):
    kyidingshipingjun=[]
    for train_index, test_index in KF.split(dataset):
        X_train, X_test = dataset[train_index], dataset[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        data, label = X_train, Y_train
        c=[]
        for i in range(len(X_test)):
            c.append(classify0(X_test[i],X_train,Y_train,k))
        #print(c)
        duile=0
        for i in range(len(Y_test)):
            if c[i] == Y_test[i]:
                duile += 1
        kyidingshipingjun.append(duile/len(Y_test))
        #上条语句计算K取值一定时，二十次k折交叉验证的平均准确率
    #下面代码给出最近邻算法的图像
    #print(kyidingshipingjun)
    # fig = plt.figure()
    # plt.title('自己编写的最近邻算法的准确度')
    # plt.xlabel('第K次10折交叉验证')
    # # 画柱形图
    # ax1 = fig.add_subplot(111)
    # ax1.bar(x=range(1,11), height=kyidingshipingjun, alpha=.7, color='g')
    # ax1.set_ylabel('准确度', fontsize='13')
    # plt.show()
    fenshu.append(sum(kyidingshipingjun)/len(kyidingshipingjun))
print('------------')
print(fenshu)
print("K取该值时准确率最高：",fenshu.index(max(fenshu))+1)
fig = plt.figure()
plt.title('自己编写的K近邻算法的准确度(K取1~50)')
plt.xlabel('K的取值')
# 画柱形图
ax1 = fig.add_subplot(111)
plt.ylim(0.5,1.0)
ax1.bar(x=range(1, 51), height=fenshu, alpha=.7, color='g')
ax1.set_ylabel('K近邻算法中K取不同值时的准确度')
plt.show()


#系统自带：
# cis=0
# KF=KFold(n_splits=10,shuffle=True)
# fenshu=[]
# for k in range(1,51):
#     kyidingshi=[]
#     for train_index, test_index in KF.split(dataset):
#         X_train, X_test = dataset[train_index], dataset[test_index]
#         Y_train, Y_test = y[train_index], y[test_index]
#         data, label = X_train, Y_train
#         from sklearn.neighbors import KNeighborsClassifier
#         knn = KNeighborsClassifier()
#         knn.fit(X_train, Y_train)
#         KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#                              metric_params=None, n_jobs=1, n_neighbors=k, p=2,
#                              weights='uniform')
#         c = knn.predict(X_test)
#         duile = 0
#         for i in range(len(Y_test)):
#             if c[i] == Y_test[i]:
#                 duile += 1
#         kyidingshi.append(duile / len(Y_test))
#     #print(len(kyidingshi))


    # fig = plt.figure()
    # plt.title('Sklearn库中最近邻算法的准确度')
    # plt.xlabel('第K次10折交叉验证')
    # # 画柱形图
    # ax1 = fig.add_subplot(111)
    # ax1.bar(x=range(1,11), height=kyidingshi, alpha=.7, color='g')
    # ax1.set_ylabel('准确度', fontsize='15')
    # plt.show()
    #fenshu.append(sum(kyidingshi)/len(kyidingshi))
print(fenshu)
fig = plt.figure()
plt.title('Sklearn库中K近邻算法的准确度(K取1~50)')
plt.xlabel('K的取值')
# 画柱形图
ax1 = fig.add_subplot(111)
plt.ylim(0.7,1.0)
ax1.bar(x=range(1, 51), height=fenshu, alpha=.7, color='g')
ax1.set_ylabel('K近邻算法中K取不同值时的准确度')
plt.show()