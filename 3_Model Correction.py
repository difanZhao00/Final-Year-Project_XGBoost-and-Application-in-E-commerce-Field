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
Xtrain=Xtrain1.head(150000)
#print(Xtrain.shape)
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
    s2.plot(title='Ratio of predition and real sales data: store {}'.format(i),figsize=(15,8))
    plt.xlabel('Shop ID')
    plt.ylabel('Ratio')
    print('Mean Ratio of predition and real sales data is {}:store {}'.format(s2['Ratio'].mean(),i))

#分析偏差最大的10个预测结果
res.sort_values(['Error'],ascending=False,inplace=True)
res[:10]

#从分析结果来看，初始模型已经可以比较好的预测保留数据集的销售趋势，相对于真实值，模型的预测值整体要偏高一些。
#从对偏差数据分析来看，偏差最大的3个数据也是明显偏高，故以保留数据集为标准对模型进行偏差校正。

#偏差整体校正优化
test=pd.read_csv('C:/XG-Boost/data/test.csv', encoding='gbk')
store=pd.read_csv('C:/XG-Boost/data/store.csv', encoding='gbk')
test.fillna(1,inplace=True)
store.fillna(0,inplace=True)
test=pd.merge(test,store,on='Store',how='left')
#print(test.info())
df_test=test.drop(['Id','Date','Open','PromoInterval'],axis=1)
print('weight correction')
W=[(0.990+(i/1000)) for i in range(20)]
S=[]
for w in W:
    error=rmspe(np.expm1(ytest),np.expm1(yhat*w))
    print('RMSPE for {:.3f}:{:.6f}'.format(w,error))
    S.append(error)
Score=pd.Series(S,index=W)
Score.plot(title='RMSPE Series Score:store {}'.format(i),figsize=(15,8))
plt.xlabel('Correction index');
plt.ylabel('RMSPE');
BS=Score[Score.values==Score.values.min()]
print('Best weight for Score:{}'.format(BS))
#当校正系数为0.994时，保留数据集的RMSPE得分最低：0.138414,相对于初始模型0.149149得分有很大的提升。
#因为每个店铺都有自己的特点，而我们设计的模型对不同的店铺偏差并不完全相同，所以我们需要根据不同的店铺进行一个细致的校正。
#细致校正：以不同的店铺分组进行细致校正，每个店铺分别计算可以取得最佳RMSPE得分的校正系数
L=range(1115)
W_ho=[]
W_test=[]
for i in L:
    s1=pd.DataFrame(res[res['Store']==i+1],columns=col_1)
    s2=pd.DataFrame(df_test[df_test['Store']==i+1])
    W1=[(0.990+(i/1000)) for i in range(20)]
    S=[]
    for w in W1:
        error=rmspe(np.expm1(s1['Sales']),np.expm1(s1['Predicition']*w))
        S.append(error)
    Score=pd.Series(S,index=W1)
    BS=Score[Score.values==Score.values.min()]
    Score.plot(title='RMSPE Series Score:store {}'.format(i),figsize=(15,8))
    plt.xlabel('Correction index');
    plt.ylabel('RMSPE');
    a=np.array(BS.index.values)
    b_ho=a.repeat(len(s1))
    b_test=a.repeat(len(s2))
    W_ho.extend(b_ho.tolist())
    W_test.extend(b_test.tolist())
    
#调整校正系数的排序
Xtest=Xtest.sort_values(by='Store')
Xtest['W_ho']=W_ho
Xtest=Xtest.sort_index()
W_ho=list(Xtest['W_ho'].values)
Xtest.drop(['W_ho'],axis=1,inplace=True)

df_test=df_test.sort_values(by='Store')
df_test['W_test']=W_test
df_test=df_test.sort_index()
W_test=list(df_test['W_test'].values)
df_test.drop(['W_test'],axis=1,inplace=True)

#计算校正后整体数据的RMSPE得分
yhat_new=yhat*W_ho
error=rmspe(np.expm1(ytest),np.expm1(yhat_new))
print('RMSPE for weight corretion {:.6f}'.format(error))
#用初始和校正后的模型对训练数据集进行预测
print('Make predictions on the test set')
from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()
df_test['StateHoliday'] = lbl.fit_transform(df_test['StateHoliday'].astype(str))
df_test['StoreType'] = lbl.fit_transform(df_test['StoreType'].astype(str))
df_test['Assortment'] = lbl.fit_transform(df_test['Assortment'].astype(str))
print(df_test.info())
dtest=xgb.DMatrix(df_test)
test_probs=gbm.predict(dtest,validate_features=False)

#初始模型
result=pd.DataFrame({'Id':test['Id'],'Sales':np.expm1(test_probs)})
result.to_csv(r'C:\XG-Boost\result\submission_1.csv',index=False)
#整体校正模型
result=pd.DataFrame({'Id':test['Id'],'Sales':np.expm1(test_probs*0.994)})
result.to_csv(r'C:\XG-Boost\result\submission_2.csv',index=False)
#细致校正模型
result=pd.DataFrame({'Id':test['Id'],'Sales':np.expm1(test_probs*W_test)})
result.to_csv(r'C:\XG-Boost\result\submission_3.csv',index=False)