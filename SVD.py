from numpy import *
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

# 相似度计算
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

def pearsSim(inA,inB):
    if len(inA) < 3:
        return 1.0
    return 0.5+0.5*corrcoef(inA,inB,rowvar = 0)[0][1]

def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

'''
用户对物品的估计评分值
dataMat:数据矩阵
user:用户编号
simMeas:相似度计算方法
item:物品编号
'''
def standEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0:
            continue
        
        # 寻找两个用户都评级的物品
        overLap = nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]

        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])
        #print("the %d and %d similarity is:%f" % (item,j,similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

# 推荐引擎
def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):
    
    # 寻找未评级的物品
    unratedItems = nonzero(dataMat[user,:].A==0)[1]
    
    if len(unratedItems) == 0:
        return "you rated everything"
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estimatedScore))

    # 寻找前N个未评级物品
    return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N]

'''
用于评分估计值
dataMat:数据集
user:用户
simMeas:相似度计算方法
item:物品项
'''
def svdEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)

    # 建立对角矩阵
    Sig4 = mat(eye(4)*Sigma[:4])
    # 构建转换后的物品
    xformedItems = dataMat.T * U[:,:4] * Sig4.I
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        print("the %d and %d similarity is:%f" % (item,j,similarity))
        simTotal += similarity
        ratSimTotal += similarity*userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

'''
图像压缩
'''
def printMat(inMat,thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1)
            else:
                print(0)
        print(" ")

def imgCompress(numSV=3,thresh=0.8):
    myl = []
    for line in open('./data/Ch14/0_5.txt'):
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    printMat(myMat,thresh)
    U,Sigma,VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV,numSV)))
    for k in range(numSV):
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print("*****reconstructed matrix using %d singular values******" % numSV)
    print(reconMat,thresh)
imgCompress(2)
exit()


print(recommend(mat(loadExData2()),1,estMethod=svdEst))
exit()
U,Sigma,VT = la.svd(mat(loadExData2()))
print(Sigma)
sig2 = Sigma**2
print(sum(sig2))
print(sum(sig2)*0.9)
print(sum(sig2[:2]))
print(sum(sig2[:3]))
exit()
myMat = mat(loadExData())
myMat[0,1] = myMat[0,0] = myMat[1,0]=myMat[2,0]=4
myMat[3,3] = 2
print(myMat)
print(recommend(myMat,2))
print(recommend(myMat,2,simMeas=ecludSim))
print(recommend(myMat,2,simMeas=pearsSim))
exit()
myMat = mat(loadExData())
print(ecludSim(myMat[:,0],myMat[:,4]))
print(ecludSim(myMat[:,0],myMat[:,0]))
print(cosSim(myMat[:,0],myMat[:,4]))
print(cosSim(myMat[:,0],myMat[:,0]))
print(pearsSim(myMat[:,0],myMat[:,4]))


'''
Data = loadExData()
U,Sigma,VT = la.svd(Data)
print(Sigma)
'''