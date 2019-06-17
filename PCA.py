from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName,delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

def pca(dataMat,topNfeat=9999999):
    meanVals = mean(dataMat,axis=0)
    
    # 取平均值
    meanRemoved = dataMat - meanVals
    
    covMat = cov(meanRemoved,rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)

    # 从小到大对N个值排序
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]

    # 将数据转换到新空间
    lowDDataMat = meanRemoved * redEigVects

    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat,reconMat

'''
将NaN替换成平均值
'''
def replaceNanWithMean():
    datMat = loadDataSet('./data/Ch13/secom.data',' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])

        # 将NaN置为平均值
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal

    return datMat

dataMat = replaceNanWithMean()
meanVals = mean(dataMat,axis=0)
meanRemoved = dataMat - meanVals
covMat = cov(meanRemoved,rowvar=0)
eigvals,eigVects = linalg.eig(mat(covMat))
print(eigvals)
exit()

dataMat = loadDataSet('./data/Ch13/testSet.txt')
lowDMat,reconMat = pca(dataMat,1)
print(shape(lowDMat))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
plt.show()
