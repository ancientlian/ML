# -*- coding: utf-8 -*-
"""
__title__ = ''
__file__ = '4决策树实现.py' 
__author__ = 'Administrator'
__mtime__ = '2019/1/24'
# 
生命中的孤独时刻只是一个人心灵的安静，
并不是一种难以忍受的负面情绪。
#
"""
import numpy as np
import pandas as pd
from math import log
import operator
########################################

"""
数据集属性标注
"""
# 年龄：0代表青年，1代表中年，2代表老年；
# 有工作：0代表否，1代表是；
# 有自己的房子：0代表否，1代表是；
# 信贷情况：0代表一般，1代表好，2代表非常好；
# 类别(是否给贷款)：no代表否，yes代表是。



"""
函数说明:创建训练数据集
Parameters:
    无
Returns:
    data - 数据集
    labels - 分类属性

"""
def creatData():
    data = [[0, 0, 0, 0, 'no'],
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return data, labels


"""
函数说明:计算给定数据集的香农熵(经验熵)
Parameters:
    dataSet - 数据集
Returns:
    shannonEnt - 香农熵(经验熵)

"""
# 香农熵：表示信息的复杂度，熵越大，则信息越复杂


def calcShannonEnt(dataSet):
    numEntires = len(dataSet)
    labelCounts = {}  # 保存每个标签(Label)出现次数的字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0  # 初始化
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntires
        shannonEnt -= prob * log(prob, 2)
    print('shannonEnt=',shannonEnt)
    return shannonEnt


"""
函数说明:按照给定特征划分数据集
Parameters:
    dataSet - 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值

"""

def splitDataSet(dataSet, axis, value):
    retDataSet = []  # 创建返回的数据集列表
    for featVec in dataSet:  # 遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 去掉axis特征
            reducedFeatVec.extend(featVec[axis + 1:])  # 将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet  # 返回划分后的数据集


"""
函数说明:按照信息增益选择最优特征
Parameters:
    dataSet - 数据集
Returns:
    bestFeature - 信息增益最大的(最优)特征的索引值

"""
# 信息增益：两个信息熵的差值，总熵-各种分类的熵值，最大的就是最好的特征分类方法

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征数
    baseEntropy = calcShannonEnt(dataSet)  # 计总香农熵
    bestInfoGain = 0.0  # 信息增益初始化
    bestFeature = -1  # 最优特征的索引值初始化
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] # 获取dataSet的第i个所有特征
        uniqueVals = set(featList)  # 创建set集合{},元素不可重复
        # print('uniqueVals=',uniqueVals)
        newEntropy = 0.0  # 经验条件熵初始化
        for value in uniqueVals:  # 计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)  # subDataSet划分后的子集
            # print('subDataSet=',subDataSet)
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy  # 信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))  # 打印每个特征的信息增益
        if (infoGain > bestInfoGain):  # 计算信息增益
            bestInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
            bestFeature = i  # 记录信息增益最大的特征的索引值
    return bestFeature

"""
函数说明:统计classList中出现次数最多的元素(类标签)
Parameters:
    classList - 类标签列表
Returns:
    sortedClassCount[0][0] - 出现此处最多的元素(类标签)
"""
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)                                                                                   #根据字典的值降序排序
    return sortedClassCount[0][0]


"""
函数说明:创建决策树
Parameters:
    dataSet - 训练数据集
    labels - 分类属性标签
    featLabels - 存储选择的最优特征标签
Returns:
    myTree - 决策树
"""
def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]  # 取分类标签(是否放贷:yes or no)

    # 如果类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 遍历完所有特征时返回出现次数最多的类标签
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选取最优特征标签
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)

    # 划分
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])  # 删除已经使用特征标签
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)  # 去重
    for value in uniqueVals:  # 遍历特征，创建决策树。
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree


"""
函数说明:使用决策树分类
Parameters:
    inputTree - 已经生成的决策树
    featLabels - 存储选择的最优特征标签
    testVec - 测试数据列表，顺序对应最优特征标签
Returns:
    classLabel - 分类结果
"""

def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))  # 获取决策树结点
    secondDict = inputTree[firstStr]  # 下一个字典
    featIndex = featLabels.index(firstStr)
    global classLabel
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel



"""
第二种分类
"""
# def classify(division_tree, feat_labels, test_vector):
#      """遍历决策树对测试数据进行分类"""
#     first_key = list(division_tree.keys())[0]
#     second_dict = division_tree[first_key]
#
#     feat_index = feat_labels.index(first_key)
#     test_key = test_vector[feat_index]
#
#     test_value = second_dict[test_key]
#
#     if isinstance(test_value, dict):
#         class_label = classify(test_value, feat_labels, test_vector)
#     else:
#         class_label = test_value
#     return class_label



"""
用树节点构造决策树
"""

if __name__ == '__main__':
    dataSet, labels = creatData()
    print("最优特征索引值:" + str(chooseBestFeatureToSplit(dataSet)))
    featLabels = []
    myTree = createTree(dataSet,labels,featLabels)
    print(myTree)
    # testVec = [2,0,0,0]
    # testVec = [0,1,1,0]
    testVec = [1,1,1,1]
    result = classify(myTree, featLabels,testVec)
    print(result)




