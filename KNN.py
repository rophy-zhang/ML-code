import operator
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

'''
创建数据集
'''
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

'''
近邻分类
参数：
inX：
dataSet：获取的样本特征集
labels: 样本数据标签
K: 要选取前k个与待测样本最近距离且已有标签的样本数据
'''
'''
numpy 知识补充：
1.shape[0]  指的是读取第一维矩阵的长度
例如： np.shape([[1,1],[2,2],[3,3]])  
       output: (3L,2L)
       其中shape[0] 就是指的矩阵有多少行，即3L
2.tile 一般是在列或者行方向上重复复制几次
例如：  np.tile([0,0],(2,1))
        output: array([[0,0],
                       [0,0]])
        指的是从行方向上重复复制2次，列1次
3.sum 一般是对矩阵一行或者一列求和
例如：  np.sum([[0,1,2],[2,1,3]],axis=1)
        output:array([3,6])
        axis=1指的是对矩阵的行进行求和
4.argsort 将数组中的元素从小到大排列，返回对应数值的索引
例如：  x = np.array([1,4,3,-1,6,9])
        x.argsort()
        output: array([3,0,2,1,4,5])
        返回对应数值的索引
5.operator.itemgetter(1) 获取对象第一个域的值
例如：  x = np.array([1,2,3,4,5])
        b = operator.itemgetter(1)
        b(x)
        output: 2
        下边用法是按照第二列值进行降序排序
'''
def classify0(inX,dataSet,labels,K):
    dataSetSize = dataSet.shape[0]

    # 距离计算  
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

    sortedDistIndicies = distances.argsort()
    classCount = {}

    # 选择最小距离的k个点
    for i in range(K):
        votelabel = labels[sortedDistIndicies[i]]
        classCount[votelabel] = classCount.get(votelabel,0) + 1

    # 排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)

    return sortedClassCount[0][0]

'''
读取文本数据，转化为numpy矩阵
'''
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)    # 得到文件行数
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    # 解析文件数据到列表
    for line in arrayOLines:
        line = line.strip()     # 去掉回车
        listFormLine = line.split('\t')
        returnMat[index,:] = listFormLine[0:3]
        classLabelVector.append(int(listFormLine[-1]))
        index += 1
    return returnMat,classLabelVector
'''
归一化特征值
'''
def autoNorm(dataSet):
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        ranges = maxVals - minVals
        normDataSet = zeros(shape(dataSet))
        m = dataSet.shape[0]
        normDataSet = dataSet - tile(minVals,(m,1))
        normDataSet = normDataSet/tile(ranges,(m,1))
        return normDataSet,ranges,minVals
'''
测试代码
'''
def datingClassTest():
    hoRatio = 0.1
    datingDataMat,datingLabels = file2matrix('./data/Ch02/datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("返回分类结果是：%d,真正类别是：%d" % (classifierResult , datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1
    print("错误率是：%f" % (errorCount/float(numTestVecs)))
'''
约会网站预测
'''
def classifyPerson():
    resultList = ['讨厌','一般','非常喜欢']
    percentTats = float(input("花费玩视频游戏占用的比例？"))
    ffMiles = float(input("每年飞行里程数？"))
    iceCream = float(input("每周消费冰激凌的公升数？"))
    datingDataMat,datingLabels = file2matrix('./data/Ch02/datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("你或许对这种人：",resultList[classifierResult-1])

'''
手写识别系统图像转向量
'''
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
'''
手写数字识别系统测试代码
'''
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('./data/Ch02/digits/trainingDigits')     # 获取目录内容
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i:] = img2vector('./data/Ch02/digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('./data/Ch02/digits/testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('./data/Ch02/digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("分类结果是：%d,真正结果是：%d" % (classifierResult,classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1
    print("分类错误数目是:%d" % errorCount)
    print("错误率：%f" % float(errorCount/mTest))



# 运行案例一
group,labels = createDataSet()
a = classify0([0,0],group,labels,3)
#print(a)


# 约会系统绘制图形
datingDataMat,datingLabels = file2matrix('./data/Ch02/datingTestSet2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
#plt.show()
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
#plt.show()
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
#plt.show()

# 约会网站案例运行
#datingClassTest()
#classifyPerson()

# 手写识别系统测试
handwritingClassTest()