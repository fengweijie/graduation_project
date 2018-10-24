# !/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
from sklearn import preprocessing
import  xgboost as xgb


def coder_columns(columns,df_date):

    coders = []
    for name in columns:
        coder = preprocessing.LabelEncoder()
        coder.fit(df_date[name].values)
        df_date[name] = coder.transform(list(df_date[name].values))
        coders.append(coder)
    return coders,df_date


if __name__ =="__main__":

    Train_Data_Set = pd.read_csv("../input/adult.data",sep=',', header=None, index_col=False)
    Test_Data_Set = pd.read_csv("../input/adult.test", sep=',', header=None, index_col=False)


    pd.set_option("display.width",400)

    # Naming the columns :
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
               'hours-per-week', 'native-country','income']

    Train_Data_Set.columns = columns
    Test_Data_Set.columns = columns

    Train_Data_Set["set"] = "train"
    Test_Data_Set["set"] =  "test"

    df_date = Train_Data_Set.append(Test_Data_Set,ignore_index=True)

    #对x数值进行编码
    coders,Train_Data_Set= coder_columns(["workclass","education","marital-status","occupation",
                                           "relationship","race","sex","native-country"],df_date)

    Train_Data_Set = df_date[df_date["set"] == "train"].reset_index(drop=True)
    Test_Data_Set = df_date[df_date["set"] == "test"].reset_index(drop=True)

    print (Train_Data_Set[:10])

    workclasssum = Train_Data_Set.groupby("workclass").size()

    #对y 进行编码
    print(Train_Data_Set[:10])
    coder = preprocessing.LabelEncoder()
    coder.fit(Train_Data_Set["income"])
    Train_Data_Set["income"] = coder.transform(list(Train_Data_Set["income"].values))
    print (Train_Data_Set[:10])

    print(Test_Data_Set[:10])
    coder = preprocessing.LabelEncoder()
    coder.fit(Test_Data_Set["income"])
    Test_Data_Set["income"] = coder.transform(list(Test_Data_Set["income"].values))
    print(Test_Data_Set[:10])

    #训练模型
    Test_Data_Set = Test_Data_Set.drop(["set"],axis=1)
    Train_Data_Set = Train_Data_Set.drop(["set"],axis=1)
    print (Test_Data_Set.dtypes)
    print (Train_Data_Set.dtypes)

    y_train = Train_Data_Set.pop("income")
    date_train = xgb.DMatrix(Train_Data_Set,label=y_train)
    y_test = Test_Data_Set.pop("income")
    date_test = xgb.DMatrix(Test_Data_Set,label= y_test)

    watch_list = [(date_test,"eval"),(date_train,"train")]
    # 开始使用模型进行训练
    # 设置参数
    # max_depth:树的最大深度,缺省值为6通常取值3-10
    # eta:为了防止过拟合,更新过程中用到的收缩步长,在每次提升计算之后,算法会直接获得新特征的权重
    # eta通过缩减特征的权重使得提升计算过程更加保守,默认值0.3  取值范围[0,1] 通常设置为[0.01-0.2]

    # silent:取0时表示打印出运行时信息，取1时表示以缄默方式运行，不打印运行时信息。缺省值为0
    # 建议取0，过程中的输出数据有助于理解模型以及调参。另外实际上我设置其为1也通常无法缄默运行

    # objective:缺省值 reg:linear 定义学习任务及相应的学习目标，可选目标函数如下：
    # “reg:linear” –线性回归。
    # “reg:logistic” –逻辑回归。
    # “binary:logistic” –二分类的逻辑回归问题，输出为概率。
    # “binary:logitraw” –二分类的逻辑回归问题，输出的结果为wTx。
    # “count:poisson” –计数问题的poisson回归，输出结果为poisson分布,在poisson回归中，max_delta_step的缺省值为0
    # “multi:softmax” –让XGBoost采用softmax目标函数处理多分类问题，同时需要设置参数num_class（类别个数）
    # “multi:softprob” –和softmax一样，但是输出的是ndata * nclass的向量，可以将该向量reshape成ndata行nclass列的矩阵。没行数据表示样本所属于每个类别的概率。
    # “rank:pairwise” –set XGBoost to do ranking task by minimizing the pairwise loss
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.01,
        'max_depth': 10,
        'missing': 0,
        'seed': 0,
        'silent': 1
    }

    bst = xgb.train(params,date_train,num_boost_round=2000,evals=watch_list,early_stopping_rounds=150)
    y_hat = bst.predict(date_test)
    print (y_hat)
    y_hat = bst.predict(date_test)
    y = date_test.get_label()
    print ('y_hat')
    print (y_hat )
    print ('y')
    print (y)
    error = sum(y != (y_hat > 0.5))
    error_rate = float(error) / len(y_hat)
    print ('样本总数：\t', len(y_hat) )
    print ('错误数目：\t%4d' % error )
    print ('错误率：\t%.5f%%' % (100 * error_rate) )
    print ("正确率：", 1 - error_rate)

