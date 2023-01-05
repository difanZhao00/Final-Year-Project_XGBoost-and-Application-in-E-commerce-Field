# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 16:36:06 2022

@author: lenovo
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from time import time
Xtrain1= pd.read_csv('C:/XG-Boost/data1/Xtrain.csv', encoding='gbk')
Xtest=pd.read_csv('C:/XG-Boost/data1/Xtest.csv', encoding='gbk')
#print(Xtrain1.shape)
Xtrain=Xtrain1.head(150000)
#print(Xtest.shape)
#拆分特征与标签，并将标签取对数处理
ytrain=np.log1p(Xtrain['Sales'])
ytest=np.log1p(Xtest['Sales'])

Xtrain=Xtrain.drop(['Sales'],axis=1)
Xtest=Xtest.drop(['Sales'],axis=1)

#定义评价函数，可以传入后面模型中替代模型本身的损失函数
def rmspe(y,yhat):
    return np.sqrt(np.mean((yhat/y-1)**2))

def rmspe_xg(yhat,y):
    y=np.expm1(y.get_label())
    yhat=np.expm1(yhat)
    return 'rmspe',rmspe(y,yhat)

#初始模型构建
#参数设定
params={'objective':'reg:linear',
       'booster':'gbtree',
       'eta':0.03,
       'max_depth':10,
       'subsample':0.9,
       'colsample_bytree':0.7,
       'seed':10}
num_boost_round=6000
dtrain=xgb.DMatrix(Xtrain,ytrain)
dvalid=xgb.DMatrix(Xtest,ytest)
watchlist=[(dtrain,'train'),(dvalid,'eval')]

#模型训练
print('Train a XGBoost model')
start=time()
gbm=xgb.train(params,dtrain,num_boost_round,evals=watchlist,
             early_stopping_rounds=100,feval=rmspe_xg,verbose_eval=True)
end=time()
#print('Train time is {:.2f} s.'.format(end-start))

#采用保留数据集进行检测
print('validating')
Xtest.sort_index(inplace=True)
ytest.sort_index(inplace=True)
yhat=gbm.predict(xgb.DMatrix(Xtest))
error=rmspe(np.expm1(ytest),np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))

#构建保留数据集预测结果
res=pd.DataFrame(data=ytest)
res['Predicition']=yhat
res=pd.merge(Xtest,res,left_index=True,right_index=True)
res['Ratio']=res['Predicition']/res['Sales']
res['Error']=abs(res['Ratio']-1)
res['Weight']=res['Sales']/res['Predicition']
res.head()

#分析保留数据集中任意三个店铺的预测结果
col_1=['Sales','Predicition']
col_2=['Ratio']
L=np.random.randint(low=1,high=1115,size=3)
print('Mean Ratio of predition and real sales data is {}:store all'.format(res['Ratio'].mean()))
for i in L:
    s1=pd.DataFrame(res[res['Store']==i],columns=col_1)
    s2=pd.DataFrame(res[res['Store']==i],columns=col_2)
    s1.plot(title='Comparation of predition and real sales data:store {}'.format(i),figsize=(15,8))
    plt.xlabel('Shop ID')
    plt.ylabel('Sales (Thousand Yuan)')
    s2.plot(title='Ratio of predition and real sales data: store {}',figsize=(15,8))
    plt.xlabel('Shop ID')
    plt.ylabel('Ratio')
    print('Mean Ratio of predition and real sales data is {}:store {}'.format(s2['Ratio'].mean(),i))
#分析偏差最大的10个预测结果
res.sort_values(['Error'],ascending=False,inplace=True)
res[:10]

#从分析结果来看，初始模型已经可以比较好的预测保留数据集的销售趋势，但相对真实值，模型的预测值整体要偏高一些。
#从对偏差数据分析来看，偏差最大的3个数据也是明显偏高。因此，我们可以以保留数据集为标准对模型进行偏差校正。
