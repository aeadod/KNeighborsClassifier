import numpy as np
import operator
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from csv import reader
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
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
        if dataset[i][4]=='setosa':
            dataset[i][4]=1
        elif dataset[i][4]=='versicolor':
            dataset[i][4]=2
        else:
            dataset[i][4]=3

filename = 'iris.csv'
dataset = load_csv(filename)
for i in range(0, len(dataset[0])-1):
     str_column_to_float(dataset, i)
str_column_to_int(dataset)
def gety(dataset):
    y=[]
    for i in dataset:
        y.append(i[4])
    return y
y=gety(dataset)
y=np.array(y)
for i in dataset:
    i.pop()
dataset=np.array(dataset)

def classify0(daice,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    #print(dataSetSize)
    diffMat = np.tile(daice,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #print(voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #print(classCount)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    #print(sortedClassCount)
    return sortedClassCount[0][0]

#自己写
cis=0
KF=KFold(n_splits=10,shuffle=True)
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


    # fig = plt.figure()
    # plt.title('自己编写的最近邻算法的准确度')
    # plt.xlabel('第K次10折交叉验证')
    # # 画柱形图
    # ax1 = fig.add_subplot(111)
    # ax1.bar(x=range(1,11), height=kyidingshipingjun, alpha=.7, color='g')
    # ax1.set_ylabel('准确度', fontsize='13')
    # plt.show()
    #print(kyidingshipingjun)

    fenshu.append(sum(kyidingshipingjun)/len(kyidingshipingjun))
print('------------')
print(fenshu)
print(fenshu.index(max(fenshu))+1)

fig = plt.figure()
plt.title('自己编写的K近邻算法的准确度(K取1~50)')
plt.xlabel('K的取值')
# 画柱形图
ax1 = fig.add_subplot(111)
plt.ylim(0.9,1.0)
ax1.bar(x=range(1, 51), height=fenshu, alpha=.7, color='g')
ax1.set_ylabel('K近邻算法中K取不同值时的准确度')
plt.show()

def showdatas(datingDataMat, datingLabels):

    fig, axs = plt.subplots(nrows=2, ncols=2)

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('green')
        if i == 2:
            LabelsColors.append('blue')
        if i == 3:
            LabelsColors.append('red')

    axs[0][0].scatter(y=datingDataMat[:,0], x=range(len(datingDataMat[:,0])), color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每种花的花萼长度',)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'花萼长度/cm')
    plt.setp(axs0_title_text, size=9, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')
    axs[0][1].scatter(x=range(len(datingDataMat[:,0])), y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每种花的花萼宽度',)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'花萼宽度/cm')
    plt.setp(axs1_title_text, size=9, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')
    axs[1][0].scatter(x=range(len(datingDataMat[:,1])), y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'每种花的花瓣长度')
    axs2_ylabel_text = axs[1][0].set_ylabel(u'花瓣长度/cm')
    plt.setp(axs2_title_text, size=9, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    #设置图例
    axs[1][1].scatter(x=range(len(datingDataMat[:, 1])), y=datingDataMat[:, 3], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs3_title_text = axs[1][1].set_title(u'每种花的花瓣宽度')
    axs3_ylabel_text = axs[1][1].set_ylabel(u'花瓣宽度/cm')
    plt.setp(axs3_title_text, size=9, weight='bold', color='black')
    plt.setp(axs3_ylabel_text, size=7, weight='bold', color='black')
    #显示图片
    plt.show()
showdatas(dataset,y)
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
#     # 准确度
#
#
#         KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#                              metric_params=None, n_jobs=1, n_neighbors=k, p=2,
#                              weights='uniform')
#         c = knn.predict(X_test)
#         #print(c)
#         duile = 0
#         for i in range(len(Y_test)):
#             if c[i] == Y_test[i]:
#                 duile += 1
#         kyidingshi.append(duile / len(Y_test))
#     print(kyidingshi)




    # fig = plt.figure()
    # plt.title('Sklearn库中最近邻算法的准确度')
    # plt.xlabel('第K次10折交叉验证')
    # # 画柱形图
    # ax1 = fig.add_subplot(111)
    # ax1.bar(x=range(1,11), height=kyidingshi, alpha=.7, color='g')
    # ax1.set_ylabel('准确度', fontsize='15')
    #     plt.show()
# fenshu.append(sum(kyidingshi)/len(kyidingshi))
# print(fenshu)
# fig = plt.figure()
# plt.title('Sklearn库中K近邻算法的准确度(K取1~50)')
# plt.xlabel('K的取值')
# # 画柱形图
# ax1 = fig.add_subplot(111)
# plt.ylim(0.9,1.0)
# ax1.bar(x=range(1, 51), height=fenshu, alpha=.7, color='g')
# ax1.set_ylabel('K近邻算法中K取不同值时的准确度')
# plt.show()
