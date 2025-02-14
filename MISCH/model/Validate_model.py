# @Author  : wxchen
"""
    micro-ROC
    在每一折中，需要分别计算 健康、三期、五期、七期的AUC，然后将四个类别的AUC平均一下，即可作为Macro-Averaged ROC
    One-vs-Rest (OvR) 方法
    原理：将多类问题转化为多个二分类问题，每次将一种类别作为正类，其余类别作为负类，分别绘制ROC曲线。
"""
import csv
import random

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, classification_report, confusion_matrix, \
    cohen_kappa_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV, GridSearchCV, ShuffleSplit, \
    StratifiedKFold
from sklearn.preprocessing import label_binarize


# 数据划分
def data_preprocessing(Abd_path,random_seed5):
    """
    The dataset was randomly divided into training (70%) and test (30%) sets,
    ensuring that samples from different scalp regions of the same subject
    were exclusively assigned to either the training or test set.
    """
    data = pd.read_csv(Abd_path)

    random.seed(int(random_seed5))
    # 同一个人的两个部位的样本不能一个训练集一个测试集
    df_healthy = data[data['AGA_severity'] == 'Healthy']
    healthy_num = []
    for i in df_healthy['num']:
        if i not in healthy_num:
            healthy_num.append(i)
    selected_num_h = random.sample(healthy_num, int(0.75 * len(healthy_num)))
    df_healthy_1 = data[data['num'].isin(selected_num_h)]

    df_type3 = data[data['AGA_severity'] == 'Type3']
    type3_num = []
    for i in df_type3['num']:
        if i not in type3_num:
            type3_num.append(i)
    selected_num_3 = random.sample(type3_num, int(0.75 * len(type3_num)))
    df_type3_1 = data[data['num'].isin(selected_num_3)]

    df_type5 = data[data['AGA_severity'] == 'Type5']
    type5_num = []
    for i in df_type5['num']:
        if i not in type5_num:
            type5_num.append(i)
    selected_num_5 = random.sample(type5_num, int(0.75 * len(type5_num)))
    df_type5_1 = data[data['num'].isin(selected_num_5)]

    df_type7 = data[data['AGA_severity'] == 'Type7']
    type7_num = []
    for i in df_type7['num']:
        if i not in type7_num:
            type7_num.append(i)
    selected_num_7 = random.sample(type7_num, int(0.75 * len(type7_num)))
    df_type7_1 = data[data['num'].isin(selected_num_7)]

    # 打印数据集1    数据集1是样本数量更多的集和 (训练集)
    df_1 = pd.concat([df_healthy_1, df_type3_1, df_type5_1, df_type7_1])
    df_2 = data.drop(df_1.index)

    return df_1, df_2



def main():

    file_path = r"D:\Github_MiSCH\MISCH\data\num_abd_all.csv"
    target = 'AGA_severity'

    features27 = ['g__Macrococcus;', 'g__Alternaria;', 'g__Malassezia;', 'g__Kocuria;', 'g__Peptoniphilus;',
                   'g__Bacteroides;', 'g__Acinetobacter;', 'g__Finegoldia;', 'g__Enhydrobacter;', 'g__Aspergillus;',
                   'g__Propionibacterium;', 'g__Moraxella;', 'g__Corynebacterium;', 'g__Anaerococcus;',
                   'g__Paracoccus;', 'g__Streptococcus;', 'g__Bartonella;', 'g__Micrococcus;', 'g__Penicillium;',
                   'g__Streptophyta_Group;', 'g__Cladosporium;', 'g__Pseudomonas;', 'g__Rhizobium;',
                   'g__Staphylococcus;', 'g__Brevundimonas;', 'g__Veillonella;', 'g__Rothia;']


    # random_seeds = np.random.randint(0, 100, size=10)
    seeds_list = [96,42,77,33,10,27,70,76,80,97]

    mean_fpr = np.linspace(0, 1, 100)
    aucs = []
    accuracy_list = []
    kappa_value_list = []

    plt.figure()

    for i in range(10):
        # Using different random seeds to partition data
        seed = seeds_list[i]
        # Generate training and testing sets through data_preprocessing
        data1, data2 = data_preprocessing(file_path, seed)

        X_train = data1[features27]
        y_train = data1[target]
        X_test = data2[features27]
        y_test = data2[target]

        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        # 计算accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_list.append(accuracy)
        # 计算kappa
        fold_kappa = cohen_kappa_score(y_test, y_pred)
        kappa_value_list.append(fold_kappa)

        # Binary the target
        y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
        n_classes = y_test_bin.shape[1]

        y_score = clf.predict_proba(X_test)

        # 计算ROC曲线和AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for j in range(n_classes):
            fpr[j], tpr[j], _ = roc_curve(y_test_bin[:, j], y_score[:, j])
            roc_auc[j] = auc(fpr[j], tpr[j])

        # 平均ROC
        all_fpr = np.unique(np.concatenate([fpr[j] for j in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for j in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[j], tpr[j])

        mean_tpr /= n_classes
        mean_auc = auc(all_fpr, mean_tpr)
        aucs.append(mean_auc)
        # 绘制ROC曲线
        plt.plot(all_fpr, mean_tpr, label=f'Fold {i + 1} (AUC = {mean_auc:.2f})')


    print('mean_accuracy', np.mean(accuracy_list))
    print('mean_kappa_value', np.mean(kappa_value_list))


    # 绘制平均ROC曲线
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('10-Fold Cross-Validation Macro ROC Curves')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    main()

