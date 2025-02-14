# @Author  : wxchen
# @Time    : 2025/2/14 15:06
"""
    MiSCH
"""
import csv
import random
from itertools import cycle
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, roc_curve, auc, accuracy_score
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV, GridSearchCV, ShuffleSplit, \
    train_test_split
from sklearn.preprocessing import label_binarize, LabelEncoder

def data_preprocessing(Abd_path, seed20):
    data = pd.read_csv(Abd_path)
    random.seed(seed20)

    df_healthy = data[data['AGA_severity'] == 'Healthy']
    healthy_num = []
    for i in df_healthy['num']:
        if i not in healthy_num:
            healthy_num.append(i)
    selected_num_h = random.sample(healthy_num, int(0.70 * len(healthy_num)))
    df_healthy_1 = data[data['num'].isin(selected_num_h)]

    df_type3 = data[data['AGA_severity'] == 'Type3']
    type3_num = []
    for i in df_type3['num']:
        if i not in type3_num:
            type3_num.append(i)
    selected_num_3 = random.sample(type3_num, int(0.70 * len(type3_num)))
    df_type3_1 = data[data['num'].isin(selected_num_3)]

    df_type5 = data[data['AGA_severity'] == 'Type5']
    type5_num = []
    for i in df_type5['num']:
        if i not in type5_num:
            type5_num.append(i)
    selected_num_5 = random.sample(type5_num, int(0.70 * len(type5_num)))
    df_type5_1 = data[data['num'].isin(selected_num_5)]

    df_type7 = data[data['AGA_severity'] == 'Type7']
    type7_num = []
    for i in df_type7['num']:
        if i not in type7_num:
            type7_num.append(i)
    selected_num_7 = random.sample(type7_num, int(0.70 * len(type7_num)))
    df_type7_1 = data[data['num'].isin(selected_num_7)]

    df_1 = pd.concat([df_healthy_1, df_type3_1, df_type5_1, df_type7_1])
    df_2 = data.drop(df_1.index)

    return df_1, df_2


def MiSCH(file_path, features27, target, all_data):
    all_predictions = []
    random.seed(29)
    random_integers = [random.randint(0, 99) for _ in range(20)]
    for i in range(20):
        main_data, sup_data = data_preprocessing(file_path,random_integers[i])
        X_train = main_data[features27]
        y_train = main_data[target]
        X_test = sup_data[features27]
        y_test = sup_data[target]

        rf_classifier = RandomForestClassifier(random_state=42)
        rf_classifier.fit(X_train,y_train)

        y_score = rf_classifier.predict_proba(X_test)

        predictions = pd.DataFrame(y_score, columns=rf_classifier.classes_)
        predictions['SampleID'] = sup_data.index
        predictions['fold'] = i + 1
        all_predictions.append(predictions)

    all_predictions = pd.concat(all_predictions)
    # 统计每个样本的预测次数
    prediction_counts = (
        all_predictions
        .groupby('SampleID')
        .size()
        .reset_index(name='prediction_count')
    )

    # 计算每个样本的平均预测概率
    average_predictions2 = (
        all_predictions
        .groupby('SampleID')
        .mean()
        .reset_index()
    )

    # 合并预测次数和平均概率
    final_results = pd.merge(average_predictions2, prediction_counts, on='SampleID')
    misch = []
    misch_pred = []
    for index, row in final_results.iterrows():
        temp_value = row['Healthy'] - 0.28 * row['Type3'] - 0.48 * row['Type5'] - 0.62 * row['Type7']
        value = 100*(temp_value+0.62) / (1+0.62)
        misch.append(value)
    final_results['misch'] = misch

    print('misch value:')
    for index, row in final_results.iterrows():
        print(row['misch'])

    for element in misch:
        if 75 < element <= 100:
            misch_pred.append('Healthy')
        elif 25 < element <= 75:
            misch_pred.append('Type3')
        elif 15 < element <= 25:
            misch_pred.append('Type5')
        elif element <= 15:
            misch_pred.append('Type7')
    final_results['misch_pred'] = misch_pred
    final_results['SampleID_copied'] = all_data['SampleID']
    final_results['AGA_severity'] = all_data['AGA_severity']

    print(final_results)

    # output_path = r'C:\Users\Administrator\Downloads\final_results.xlsx'
    # final_results.to_excel(output_path, index=False)


def main():
    file_path = r"D:\Github_MiSCH\MISCH\data\num_abd_all.csv"
    all_data = pd.read_csv(file_path, usecols=lambda x: x != 'num')
    target = 'AGA_severity'
    features = ['g__Macrococcus;', 'g__Alternaria;', 'g__Malassezia;', 'g__Kocuria;', 'g__Peptoniphilus;',
                   'g__Bacteroides;', 'g__Acinetobacter;', 'g__Finegoldia;', 'g__Enhydrobacter;', 'g__Aspergillus;',
                   'g__Propionibacterium;', 'g__Moraxella;', 'g__Corynebacterium;', 'g__Anaerococcus;',
                   'g__Paracoccus;', 'g__Streptococcus;', 'g__Bartonella;', 'g__Micrococcus;', 'g__Penicillium;',
                   'g__Streptophyta_Group;', 'g__Cladosporium;', 'g__Pseudomonas;', 'g__Rhizobium;',
                   'g__Staphylococcus;', 'g__Brevundimonas;', 'g__Veillonella;', 'g__Rothia;']

    MiSCH(file_path,features,target,all_data)

if __name__ == '__main__':
    main()
