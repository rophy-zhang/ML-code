from numpy import *

def predict(w,x):
    return w*x.T

'''
T:迭代次数
k:待处理列表的大小
'''
def batchPegasos(dataSet,labels,lam,T,k):
    m,n = shape(dataSet)
    w = zeros(n)
    dataIndex = range(m)
    for t in range(1,T+1):
        wDelta = mat(zeros(n))
        eta = 1.0/(lam*t)     # 学习率
        random.shuffle(dataIndex)
        for j in range(k):
            i = dataIndex[j]
            p = predict(w,dataSet[i,:])
            if labels[i]*p < 1:
                wDelta += labels[i]*dataSet[i,:].A
        w = (1.0 - 1/t)*w + (eta/k)*wDelta
    return w