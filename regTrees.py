from numpy import *
from tkinter import *
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

'''
加载文件数据
'''
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        
        # 将每行映射称为浮点数
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

'''
根据当前特征值切分数据
dataSet:数据集合
feature:带切分特征
value:改特征的某个值
'''
def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0,mat1

'''
创建叶节点
'''
def regLeaf(dataSet):
    return mean(dataSet[:,-1])

'''
误差计算
'''
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

'''
选择最优划分特征
ops:用于控制函数的停止时机
'''
def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS = ops[0]       # 容许的误差下降值
    tolN = ops[1]       # 切分的最少样本数
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None,leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None,leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None,leafType(dataSet)
    return bestIndex,bestValue

'''
dataSet:数据集
leafType：建立叶节点的函数
errType：误差计算函数
ops：一个包含树结构所建所需其他参数的元组
'''
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

'''
判断当前节点是不是一颗树
'''
def isTree(obj):
    return (type(obj).__name__=='dict')

'''
获取树合并的节点均值
'''
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2.0

'''
后剪枝函数
tree:待剪枝的树
testData：剪枝所需的测试数据
'''
def prune(tree,testData):
    if shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge   = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:
            print("合并")
            return treeMean
        else:
            return tree
    else:
        return tree

'''
格式化数据集，求解w
'''
def linearSolve(dataSet):
    m,n = shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError("这个矩阵是奇异矩阵，不可以求逆,\n 试着增加ops倍数")
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

'''
获取模型权重
'''
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

'''
计算线性模型误差
'''
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X*ws
    return sum(power(Y-yHat,2))


def regTreeEval(model,inDat):
    return float(model)

def modelTreeEval(model,inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1] = inDat
    return float(X*model)


def treeForeCast(tree,inData,modelEval = regTreeEval):
    if not isTree(tree):
        return modelEval(tree,inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)

def createForeCast(tree,testData,modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree,mat(testData[i]),modelEval)
    return yHat

#-------------------------------------------构建图形界面----------------------------------------

def reDraw(tolS,tolN):
    reDraw.f.clf()  #清空之前的图像
    reDraw.a = reDraw.f.add_subplot(111)#重新添加新图
    if chkBtnVar.get():#检查选框model tree是否被选中
        if tolN < 2: tolN = 2
        myTree = createTree(reDraw.rawDat, modelLeaf,modelErr, (tolS, tolN))
        yHat = createForeCast(myTree, reDraw.testDat, modelTreeEval)
    else:
        myTree = createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = createForeCast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:, 0], reDraw.rawDat[:, 1], s=5)  # 绘制真实值
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)  # 绘制预测值
    reDraw.canvas.show()

def getInputs():#获取输入
    try:#期望输入是整数
        tolN = int(tolNentry.get())
    except:#清楚错误用默认值替换
        tolN = 10
        print("enter Integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')
    try:#期望输入是浮点数
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter Float for tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS

def drawNewTree():
    tolN, tolS = getInputs()  # 从输入文本框中获取参数
    reDraw(tolS, tolN)  #绘制图


root = Tk()

reDraw.f = Figure(figsize=(5, 4), dpi=100)  # 创建画布
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Label(root, text="tolN").grid(row=1, column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')
Label(root, text="tolS").grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')
Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text="Model Tree", variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.rawDat = mat(loadDataSet('sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
reDraw(1.0, 10)

root.mainloop()



# 测试
myDat = loadDataSet('./data/Ch09/ex00.txt')
myMat = mat(myDat)
#print(createTree(myMat))
myDat1 = loadDataSet('./data/Ch09/ex0.txt')
myMat1 = mat(myDat1)
#print(createTree(myMat1))

myDat2 = loadDataSet('./data/Ch09/ex2.txt')
myMat2 = mat(myDat2)
#myTree = createTree(myMat2)
myTree = createTree(myMat2,ops=(0,1))
#print(myTree)
myDatTest = loadDataSet('./data/Ch09/ex2test.txt')
myMat2Test = mat(myDatTest)
#trees = prune(myTree,myMat2Test)

#print(trees)
myDat3 = loadDataSet('./data/Ch09/exp2.txt')
myMat3 = mat(myDat3)
trees = createTree(myMat3,modelLeaf,modelErr,(1,10))
#print(trees)


# 骑自行车速度与智力测试
trainMat = mat(loadDataSet('./data/Ch09/bikeSpeedVsIq_train.txt'))
testMat  = mat(loadDataSet('./data/Ch09/bikeSpeedVsIq_test.txt'))
myTree = createTree(trainMat,ops=(1,20))
yHat = createForeCast(myTree,testMat[:,0])
rt = corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
print(rt)

myTree = createTree(trainMat,modelLeaf,modelErr,(1,20))
yHat = createForeCast(myTree,testMat[:,0],modelTreeEval)
rt = corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
print(rt)

ws,X,Y = linearSolve(trainMat)
for i in range(shape(testMat)[0]):
    yHat[i] = testMat[i,0]*ws[1,0]+ws[0,0]
rt = corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
print(rt)

