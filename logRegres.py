import numpy as np
import matplotlib.pyplot as plt


'''
文本数据转化为矩阵
'''
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('./data/Ch05/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

'''
定义sigmoid函数
inX: 关系式的结果
'''
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

'''
梯度上升
dataMatIn: 数据矩阵
classLabels: 标签矩阵
'''
def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose() # 转置操作
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose()*error
    return weights

'''
画出拟合直线
wei: 权重
'''
def plotBestFit(weights):
    dataMat,labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = np.arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]  # W0X0+W1X1+W2X2 = 0的变形，X0=1
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# 随机梯度上升算法
def stocGradAscent0(dataMatrix,classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(2000):
        for i in range(m):
            h = sigmoid(np.sum(dataMatrix[i]*weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]
    return weights

# 改进随机梯度上升算法
def stocGradAscent1(dataMatrix,classLabels,numIter = 150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.01      # 每次迭代调整alpha
            # 随机选取样本更新权重
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

'''
Logistic回归分类函数
inX: 待测样本
weights: 训练后的权重
'''
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
'''
训练测试函数
'''
def colicTest():
    frTrain = open('./data/Ch05/horseColicTraining.txt') 
    frTest = open('./data/Ch05/horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet),trainingLabels,500)
    errorCount = 0
    numTetsVec = 0.0
    for line in frTest.readlines():
        numTetsVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTetsVec)
    print("错误率是:%f" % errorRate)
    return errorRate

'''

'''
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("这%d次的平均错误率是:%f" % (numTests,errorSum/float(numTests)))
    
# 测试
#dataArr,labelMat = loadDataSet()
#weights = gradAscent(dataArr,labelMat)
#plotBestFit(weights.getA())
#weights = stocGradAscent1(np.array(dataArr),labelMat,500)
#plotBestFit(weights)

# 从氙气病症预测病马的死亡率
multiTest()