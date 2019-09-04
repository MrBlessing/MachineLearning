from math import log

#计给定数据集的香农熵
def calcShannonEnt(dataSet):
    m = len(dataSet)
    #统计每个label出现的次数
    labelsCount = {}
    for i in dataSet:
        label = i[-1]
        labelsCount[label] = labelsCount.get(label,0)+1
    shannonEnt = 0.0
    for key in labelsCount:
        #该分类的概率
        p = float(labelsCount[key])/m
        #对每个分类求熵并累加
        shannonEnt += -1*log(p,2)*p
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ["no surfacing",'flippers']
    return dataSet,labels

#划分数据集
#第一个参数是要划分的数据集，第二个参数是划分的位置，第三个参数是需要划分位置的值
def splitDataSet(dataSet,index,value):
    res = [];
    for i in dataSet:
        if i[index] == value:
            temp = i.copy()
            del temp[index]
            res.append(temp)
    return res

#寻找到最好的数据集分割方法
def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0])-1
    #未分割前的香农熵
    baseEnt = calcShannonEnt(dataSet)
    #最合适的信息增益
    bestInfoGain = -1;
    bestFeature = 0
    for i in range(numFeature):
        #取出每一行的元素
        col = [e[i] for e in dataSet]
        #取出当中包含的不重复的特征值
        colSet = set(col)
        for value in colSet:
            #用index=i来划分数据集
            subDataSet = splitDataSet(dataSet,i,value)
            #按照划分的比例计算划分之后的信息熵
            newEnt = len(subDataSet)/float(len(dataSet))*calcShannonEnt(subDataSet)
        #计算信息增益
        infoGain = baseEnt-newEnt
        #取信息增益最大者
        if (infoGain>bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i 
    return bestFeature

#创造决策树
def createTree(dataSet,labels):
    #如果划分的数据中标签都相同
    classList = [e[-1] for e in dataSet]
    if(len(set(classList))==1):
        return classList[0]
    #如果正在遍历最后一个特征值，返回出现最多的标签
    if(len(dataSet[0])==1):
        return majorityCnt(classList)
    #选择最合适的特征值进行划分
    bestFeat = chooseBestFeatureToSplit(dataSet)
    #该特征值对应的标签
    bestLabel = labels[bestFeat]
    #根据标签构建一个节点
    Tree = {bestLabel:{}}
    #标签加入树后进行删除
    labels.remove(bestLabel)
    #取出该列特征值中所有不同元素
    featSet = set([e[bestFeat] for e in dataSet])
    #依次遍历该特征变量的不同值
    for i in featSet:
        #子标签
        subLabels = labels[:]
        Tree[bestLabel][i] = createTree(splitDataSet(dataSet,bestFeat,i),subLabels)
    return Tree

#取出数据集中出现最多的标签
def majorityCnt(classList):
    classCount = {}
    for i in classList:
        classCount[i] = classCount.get(i,0)+1
    sortClass = sorted(classCount,key=classCount.get,reverse=True)
    return sortClass[0]

#根据决策树进行决策
#参数3为需要进行分类得向量列表，向量的顺序默认和训练集向量顺序一致
def classify(tree,testlabels,testVec):
    #取得树中得第一个key
    firstKey = list(tree.keys())[0]
    #找到标签中第一个节点的索引
    firstIndex = testlabels.index(firstKey)
    #该标签对应的值
    value = tree[firstKey];
    #开始遍历该标签下的节点值
    for key in value.keys() :
        if testVec[firstIndex] == key:
            if type(value[key]).__name__=="dict":
                classLabel = classify(value[key],testlabels,testVec)
            else : classLabel = value[key];
    return classLabel