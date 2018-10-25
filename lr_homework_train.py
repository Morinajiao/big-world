# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:10:29 2018

@author: asus
"""

import pandas as pd
import numpy as np
import copy

file = pd.read_csv('train.csv')
feature_df = pd.DataFrame()
label_df = file['Label']
columns = file.columns.values.tolist()
for i in range(2,15):
    tmp_se = file[columns[i]]
    num = copy.deepcopy(tmp_se).dropna().mean()
    tmp_se = tmp_se.fillna(num)
    feature_df = pd.concat([feature_df, tmp_se], axis = 1)
b_df = pd.Series([1] * len(feature_df))
feature_df = pd.concat([b_df, feature_df], axis = 1)
feature = np.mat(feature_df)
label = np.mat(label_df).T



def load_data(file_name):
    '''导入训练数据
    input:  file_name(string)训练数据的位置
    output: feature_data(mat)特征
            label_data(mat)标签
    '''
    f=open(file_name)
    feature_data=[]
    label_data=[]
    for line in f.readlines():
        feature_tmp=[]
        label_tmp=[]
        lines=line.strip().split("\t")
        feature_tmp.append(1)   #偏置项
        for i in range(len(lines)-1):
            feature_tmp.append(float(lines[i]))
        label_tmp.append(float(lines[-1]))
        
        feature_data.append(feature_tmp)
        label_data.append(label_tmp)
    f.close()
    return np.mat(feature_data),np.mat(label_data)

def sig(x):
    return 1.0/(1+np.exp(-x))




def Ir_train_bgd(feature,label,maxCycle,alpha):
    '''feature是特征
       label是标签
       maxCycle是最大迭代次数
       alpha是学习率，也叫做步长
    '''
    n=np.shape(feature)[1]    #特征个数取列数也就是3
    w=np.mat(np.ones((n,1)))  #初始化权重全1矩阵3*1
    i=0
    while i<= maxCycle:
        i+=1
        h=sig(feature*w)     #计算判断归属样本的概率 200*1
        err=label-h          #y(i)-sig(WX(i)+b)  200*1
        if i%100 ==0:
            print("--------iter"+str(i)+",错误率="+str(error_rate(h,label)))
        #修正权重。对的，求导后是y(i)-sig(WX(i)+b)乘上x(i)在求均值
        w=w+alpha*feature.T*err/np.shape(feature)[0]  
    return w

def error_rate(h,label):
    m=np.shape(h)[0]
    sum_err=0.0
    for i in range(m):
        if h[i]>0 and (1-h[i])>0:
            #损失函数[y(i)log(h)+(1-y(i))log(1-h)]/m
            sum_err -= (label[i,0]*np.log(h[i,0])+(1-label[i,0])*np.log(1-h[i,0]))
        else:
            sum_err-=0
        return sum_err/m

def save_model(file_name,w):
    m=np.shape(w)[0]
    f_w=open(file_name,"w")
    w_array=[]
    for i in range(m):
        w_array.append(str(w[i,0]))
    f_w.write("\t".join(w_array))   #将w_arrary的字符串以中间tab来连接
    f_w.close()
    


   
#print("-------训练样本-------")
#w=Ir_train_bgd(feature,label,1000,0.01)
#print("-------保存模型-------")
#save_model("weights",w)




from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression()  # 使用类，参数全是默认的
classifier.fit(feature, label)  # 训练数据来学习，不需要返回值
 
file_test = pd.read_csv('test.csv')
feature_df_test = pd.DataFrame()
columns = file_test.columns.values.tolist()
for i in range(1,14):
    tmp_se = file_test[columns[i]]
    num = copy.deepcopy(tmp_se).dropna().mean()
    tmp_se = tmp_se.fillna(num)
    feature_df_test = pd.concat([feature_df_test, tmp_se], axis = 1)
b_df = pd.Series([1] * len(feature_df_test))
feature_df_test = pd.concat([b_df, feature_df_test], axis = 1)
feature_test = np.mat(feature_df_test)

x = classifier.predict(feature_test)  # 测试数据，分类返回标记

file1 = pd.read_csv('submission.csv')
result_df = pd.DataFrame()
result_df = pd.concat([result_df, file1['Id']], axis = 1)
result_df = pd.concat([result_df, pd.Series([int(i) for i in x.T.tolist()])], axis = 1)
result_df = pd.concat([result_df, file1['Label']], axis = 1)

count = 0
for i in range(len(result_df)):
    if result_df.loc[i,0] != result_df.loc[i,'Label']:
        count += 1       #计算误差








