from numpy import *
#操作符库
import operator
from os import listdir

#产生数据
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#k-近邻算法
#inX为输入向量
def classify0(inX,dataSet,labels,k):
    #得到数据集的行数
    dataSetSize = dataSet.shape[0]
    #把输入向量的行数扩充成和数据集的行数一致，计算两者之间的差值
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    #差值进行平方
    sqDiffMat = diffMat**2
    #将1维的数字加和,开根号
    distances = sqDiffMat.sum(axis=1)**0.5
    #将其中的数据进行排序，获得排序好的序号
    indexs = distances.argsort();
    classcount = {}
    for i in range(k):
        voteILabel = labels[indexs[i]]
        #分别统计前k个数据的分类频率
        classcount[voteILabel] = classcount.get(voteILabel,0)+1
    #将分类的频率进行排序，取第一个（也就是对值进行排序,从大到小排序）,该函数返回的是一个列表
    sortedClassCount = sorted(classcount,key=classcount.get,reverse=True)
    return sortedClassCount[0]

def file2matrx(fileName):
    file = open(fileName)
    lines = file.readlines()
    length = len(lines)
    group = zeros((length,3))
    labels=[]
    #记录正在处理的行数
    index = 0
    for i in lines:
        #去掉头尾的空格和换行符
        i = i.strip();
        eachElem = i.split('\t')
        group[index,:] = eachElem[0:3]
        labels.append(int(eachElem[3]))
        index+=1
    return group,labels

#归一化特征值
def autoNorm(dataSet):
    minValue = dataSet.min(0)
    maxValue = dataSet.max(0)
    normDataset = zeros(shape(dataSet))
    #总的行数
    m = dataSet.shape[0]
    normDataset = (dataSet-tile(minValue,(m,1)))/tile(maxValue-minValue,(m,1))
    return normDataset,minValue,maxValue

#分类效果测试
def datingClassTest():
    group,labels = file2matrx("datingTestSet2.txt")
    norm,minV,maxV = autoNorm(group)
    m =norm.shape[0]
    #划定出10%为测试集
    m = group.shape[0]
    testVec = int(0.1*m)
    error = 0
    for i in range(testVec):
        result = classify0(norm[i,:],norm[testVec:m],labels[testVec:m],3)
        print("the classfier came back with : %d the real answer is : %d" % (result,labels[i]))
        if(result!=labels[i]): error += 1
    print("the total error rate is : %f" % (error/float(testVec)))

#读取文件内容转化成数组的形式
def image2vector(fileName):
    matrx = zeros((1,1024))
    file = open(fileName)
    for i in range(32):
        line = file.readline()
        for j in range(32):
            matrx[0,32*i+j] = int(line[j])
    return array(matrx)

#识别文件中的数字
def handWritingClassTest():
    trainLabels = []
    #从文件名中取得标签
    fileList = listdir("trainingDigits")
    m = len(fileList)
    #训练集的特征值数组
    trainSet = zeros((m,1024))
    for i in range(m):
        label = fileList[i].split("_")[0]
        trainLabels.append(int(label))
        trainSet[i,:] = image2vector("trainingDigits/%s" % fileList[i])
    #处理测试集的数据
    fileList = listdir("testDigits")
    m = len(fileList)
    testSet = zeros((m,1024))
    error = 0
    for i in range(m):
        #真实的labels
        trueLabel = int(fileList[i].split("_")[0])
        testSet = image2vector("trainingDigits/%s" % fileList[i])
        #利用该算法估计标签
        testLabel = classify0(testSet,trainSet,trainLabels,10)
        print("the classifier came back with : %d, the real answer is : %d " % (testLabel,trueLabel))
        if(testLabel!=trueLabel): error+=1
    print("\nthe total error rate is : %f" % (error/float(m)))

handWritingClassTest()