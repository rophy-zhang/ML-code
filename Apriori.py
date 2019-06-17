'''
创建简单数据集
'''
def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

'''
构建大小为1所有候选项的集合
'''
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset,C1))

'''
D:数据集
Ck:包含候选集合的列表
minSupport：感兴趣项的最小支持度
'''
def scanD(D,Ck,minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid): # 判断候选项中是否含数据集的各项
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList,supportData

'''
Lk:频繁项集列表
k：项集元素个数
输出：Ck
'''
def aprioriGen(Lk,k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):

        # 前k-2个项相同时，将两个集合合并
        for j in range(i+1,lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i]|Lk[j])

    return retList

def apriori(dataSet,minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set,dataSet))
    L1,supportData = scanD(D,C1,minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2],k)
        
        # 扫描数据集，从Ck得到Lk
        Lk,supK = scanD(D,Ck,minSupport)

        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L,supportData

def calcConf(freqSet,H,supportData,brl,minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet,H,supportData,brl,minConf=0.7):
    m = len(H[0])
    if len(freqSet) > (m + 1):
        Hmp1 = aprioriGen(H,m+1)
        Hmp1 = calcConf(freqSet,Hmp1,supportData,brl,minConf)
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet,Hmp1,supportData,brl,minConf)
'''
L:频繁项集列表
supportData:频繁项集支持数据的字典
minConf:最小可信度阈值
'''            
def generateRules(L,supportData,minConf=0.7):
    bigRuleList = []
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList 

mushDatSet = [line.split() for line in open('./data/Ch11/mushroom.dat').readlines()]
L,suppData = apriori(mushDatSet,minSupport=0.3)
for item in L[1]:
    if item.intersection('2'):
        print(item)
for item in L[2]:
    if item.intersection('2'):
        print(item)
for item in L[3]:
    if item.intersection('2'):
        print(item)

#dataSet = loadDataSet()
#C1 = createC1(dataSet)
#D = list(map(set,dataSet))
#L1,suppData0 = scanD(D,C1,0.5)
#L,suppData = apriori(dataSet,minSupport=0.5)
#rules = generateRules(L,suppData,minConf=0.5)
#print(rules)