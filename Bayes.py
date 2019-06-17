import numpy as np
import re
import feedparser
import operator

'''
词表到向量转换函数
'''
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']] 
    classVec = [0,1,0,1,0,1]    # 1代表侮辱性文字，0代表正常言论
    return postingList,classVec
'''
获取文档中的词汇表
'''
def createVocabList(dataSet):
    vocabSet = set([])      # 创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)     # 创建两个集合的并集
    return list(vocabSet)

'''
文档转化为向量
vocabList： 词汇列表
inputSet： 输入文档集
'''
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("词： %s 不在我的词汇表！" % word)
    return returnVec

'''
朴素贝叶斯词袋模型
'''
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

'''
朴素贝叶斯分类器训练函数
trainMatrix: 文档矩阵
trainCategory： 每篇文档类别标签向量
'''
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)  #3/6=0.5
    
    # 初始化概率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:

            # 向量相加
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    
    # 对每个元素做除法
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)

    return p0Vect,p1Vect,pAbusive

'''
朴素贝叶斯分类函数
vec2Classify: 要分类的向量
p0Vec: 0标签条件概率
p1Vec： 1标签条件概率
pClass1: 类别1概率
'''
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))

'''
文本解析
'''
def textParse(bigString):
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

'''
垃圾邮件测试文件
'''
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList = textParse(open('./data/Ch04/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('./data/Ch04/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print('分类错误的测试集：',docList[docIndex])
    print("错误率：",float(errorCount)/len(testSet))

'''
RSS源分类器
'''
def calcMostFreq(vocabList,fullText):
    # 计算出现频率
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)

    sortedFreq = sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]

'''
高频词去除函数
'''
def localWords(feed1,feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen))
    testSet = []
    for i in range(20):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print("错误率是：",float(errorCount)/len(testSet))
    print('分类错误的测试集：',docList[docIndex])
    return vocabList,p0V,p1V

'''
最具有表征性的词汇显示
'''
def getTopWords(ny,sf):
    vocabList,p0V,p1V = localWords(ny,sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -0.6 :
            topSF.append((vocabList[i]),p0V[i])
        if p1V[i] > -0.6 :
            topNY.append(vocabList[i],p1V)
    sortedSF = sorted(topSF,key=lambda pair:pair[1],reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY,key=lambda pair:pair[1],reverse=True)
    print("NF*NF*NF*NF*NF*NF*NF*NF*NF*NF*NF*NF*NF*NF*NF*NF*")
    for item in sortedNY:
        print(item[0])

# 测试实例
#listOPosts,listClasses = loadDataSet()
#myVocabList = createVocabList(listOPosts)
# print(myVocabList)
#print(setOfWords2Vec(myVocabList,listOPosts[0]))
#trainMat = []
#for postinDoc in listOPosts:
 #   trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
#p0V,p1V,pAb = trainNB0(trainMat,listClasses)
#print(p0V)
#print(p1V)
#print(pAb)
#testingNB()


# 垃圾邮件过滤测试
#spamTest()

# 从个人广告中获取区域倾向
ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
sf = feedparser.parse('http://rss.tom.com/happy/happy.xml')
vocabList,pSF,pNY = localWords(ny,sf)
getTopWords(ny,sf)
