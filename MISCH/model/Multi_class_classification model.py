# @Author  : Xiaochen Wang
"""
    通过微生物数据对人类头部健康or脱发进行预测
"""
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV, GridSearchCV, ShuffleSplit
from sklearn.preprocessing import label_binarize


def multi_kingdom(path_1,path_2):
    with open(path_1,'r') as file1:
        header1 = file1.readline().strip().split('\t')[1:]

    with open(path_2,'r') as file2:
        header2 = file2.readline().strip().split('\t')[1:]
    print(f'header1',header1)
    print(f'header2',header2)
    return header1, header2

def data_preprocessing(Abd_path):
    data = pd.read_csv(Abd_path)
    return data

def feature_importance(data,target):

    feature_order = []
    importance_order = []

    X = data.iloc[:,2:]
    y = data[target]
    importance_list = []
    kf = KFold(n_splits=10,shuffle=True,random_state=9)

    for train_index, _ in kf.split(X):
        X_train, y_train= X.iloc[train_index], y.iloc[train_index]
        model = RandomForestClassifier(random_state=9)
        model.fit(X_train, y_train)
        feature_importances = model.feature_importances_

        importance_list.append(feature_importances)

    average_importance = np.mean(importance_list,axis=0)
    feature_names = X.columns
    # 将每个物种的物种名称和对应的特征重要性放入字典中
    feature_importances_dict = {feature_name:importance for feature_name,
                                importance in zip(feature_names,average_importance)}
    sorted_features = sorted(feature_importances_dict.items(),key=lambda x: x[1],reverse=True)

    for feature,importance in sorted_features:
        feature_order.append(feature)
        importance_order.append(importance)

    return feature_order

def best_features(features_list,data,target):

    features = data.iloc[:,2:]
    target = data[target]
    AGA_types = target.unique()
    n_class = AGA_types.size    # 疾病标签个数
    target_code = pd.Categorical(target).codes  # 将健康、脱发三期、脱发五期、脱发七期转换为0/1/2/3数字
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    kappa_10 = []

    for i in range(1, len(features_list) + 1):
        selected_features = features_list[:i]
        cv10_kappa = []
        for train_index,test_index in kf.split(features):
            X_train,y_train = features[selected_features].iloc[train_index], target_code[train_index]
            X_test,y_test = features[selected_features].iloc[test_index], target_code[test_index]
            y_one_hot = label_binarize(y_test, classes=np.arange(n_class))   #转换为二进制编码

            rf_model = RandomForestClassifier(random_state=9)
            rf_model.fit(X_train,y_train)
            y_pred = rf_model.predict(X_test)
            kappa = cohen_kappa_score(y_test, y_pred)
            cv10_kappa.append(kappa)
        kappa_10.append(np.mean(cv10_kappa))

    for i in range(len(kappa_10)):
        plt.scatter(i + 1, kappa_10[i], color='#B3B3B3')
    for i in range(len(kappa_10)-1):
        plt.plot([i + 1, i + 2], [kappa_10[i], kappa_10[i + 1]], color='#B3B3B3')

    min_features_count = np.argmax(kappa_10) + 1
    optimal_features = features_list[:min_features_count]
    print(f'optimal_features:{optimal_features},optimal_features count:{len(optimal_features)}')

    # bacteria_patch = mpatches.Patch(color='#3a4892', label='Bacteria')
    # fungi_patch = mpatches.Patch(color='#ed0000', label='Fungi')
    # plt.legend(handles=[bacteria_patch, fungi_patch])
    plt.xlabel('features count')
    plt.ylabel('cv10_accuracy')
    plt.title('Model Performance with Increasing Features')
    plt.show()

    return optimal_features

def model_parameters(data,features,target):

    features_data = data[features]
    target_data = data[target]
    rf_model = RandomForestClassifier()
    n_estimators = [int(x) for x in np.linspace(start=50, stop=1000, num=20)]
    min_samples_split = [2, 5, 8]
    min_samples_leaf = [1, 2, 4]
    max_features = [None, 'log2', 'sqrt']
    max_depth = [8, 10, 15, 20]

    random_param_group = {'n_estimators':n_estimators,
                          'min_samples_split':min_samples_split,
                          'min_samples_leaf':min_samples_leaf,
                          'max_features': max_features,
                          'max_depth':max_depth,
                          }
    rf_model = RandomizedSearchCV(rf_model,
                                  param_distributions=random_param_group,
                                  n_iter=200,
                                  scoring='roc_auc_ovr',
                                  cv=10,
                                  verbose=2,
                                  n_jobs=-1,
                                  random_state=42)
    rf_model.fit(features_data,target_data)

    print("Best Parameters: ", rf_model.best_params_)
    print("Best Score: ", rf_model.best_score_)


    param_grid = {
        "n_estimators":[int(x) for x in np.linspace(start = 200,stop = 400,num = 9)],
        "min_samples_split":[2,3],
        "min_samples_leaf":[3,4,5],
        "max_features": [None,'log2','sqrt'],
        "max_depth":[10,12,15,18]}
    # 小范围穷举搜索最佳参数组合
    # grid = GridSearchCV(rf_model,param_grid,scoring='roc_auc_ovr',cv=5,n_jobs=-1)
    # grid.fit(features_data,target_data)
    # print('GridSearchCV策略')
    # print("grid Best Parameters: ", grid.best_params_)
    # print("grid Best Score: ", grid.best_score_)
    # grid_parameters_dict = {'max_depth': 10,
    #                         'max_features': 'sqrt',
    #                         'min_samples_leaf': 3,
    #                         'min_samples_split': 3,
    #                         'n_estimators': 300}

    # return grid_parameters_dict

def AGA_pred_model(data,features,target,parameters_dict):

    features_data = data[features]
    target_data = data[target]
    AGA_types = target_data.unique()
    n_class = AGA_types.size
    target_code = pd.Categorical(target_data).codes

    # rf_model = RandomForestClassifier(**parameters_dict,bootstrap=True,random_state=42)
    rf_model = RandomForestClassifier(random_state=42)

    shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    roc_auc_scores = []

    for train_index, test_index in shuffle_split.split(features_data):
        X_train, y_train = features_data.iloc[train_index], target_code[train_index]
        X_test, y_test = features_data.iloc[test_index], target_code[test_index]
        y_one_hot = label_binarize(y_test, classes=np.arange(n_class))

        rf_model.fit(X_train, y_train)
        y_pred_proba = rf_model.predict_proba(X_test)
        print(y_pred_proba)
        auc_micro = roc_auc_score(y_test,y_pred_proba,multi_class='ovr',average='micro')
        print('auc_micro:',auc_micro)
        roc_auc_scores.append(auc_micro)

    print('roc_auc_scores', roc_auc_scores)
    print('np.mean(roc_auc_scores):', np.mean(roc_auc_scores))
    return rf_model

def AGA_pred(AGA_pred_model, scalp_data, features, target):
    features_data = scalp_data[features]
    predicted_data = AGA_pred_model.predict(features_data)
    print(predicted_data)

    degree_labels = ['Healthy', 'Type3', 'Type5', 'Type7']
    y_pred_proba = AGA_pred_model.predict_proba(features_data)
    print(y_pred_proba)

    predicted_labels = []
    for i in range(len(y_pred_proba)):
        if y_pred_proba[i,0]>0.5:
            predicted_labels.append('Healthy')
        else:
            max_prob_index = np.argmax(y_pred_proba[i,1:])+1
            predicted_labels.append(degree_labels[max_prob_index])
    print('predicted_labels')
    print(predicted_labels)

    accuracy = np.mean(np.array(predicted_labels) == np.array(scalp_data[target]))


def main():

    file_path = r"D:\Github_MiSCH\MISCH\data\multi_class_abd_all.csv"

    target = 'AGA_severity'
    data = data_preprocessing(file_path)

    # features_list = feature_importance(data,target)
    # print('feature_list:',features_list)
    # features = best_features(features_list,data,target)

    features27 = ['g__Macrococcus;', 'g__Alternaria;', 'g__Malassezia;', 'g__Kocuria;', 'g__Peptoniphilus;',
                   'g__Bacteroides;', 'g__Acinetobacter;', 'g__Finegoldia;', 'g__Enhydrobacter;', 'g__Aspergillus;',
                   'g__Propionibacterium;', 'g__Moraxella;', 'g__Corynebacterium;', 'g__Anaerococcus;',
                   'g__Paracoccus;', 'g__Streptococcus;', 'g__Bartonella;', 'g__Micrococcus;', 'g__Penicillium;',
                   'g__Streptophyta_Group;', 'g__Cladosporium;', 'g__Pseudomonas;', 'g__Rhizobium;',
                   'g__Staphylococcus;', 'g__Brevundimonas;', 'g__Veillonella;', 'g__Rothia;']

    # parameters_dict = model_parameters(data,features27,target)
    parameters_dict = {'n_estimators': 600, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 15}

    AGA_model = AGA_pred_model(data,features27,target,parameters_dict)

    # pred_data_1 = AGA_pred(AGA_model,data_1,features27,target)


if __name__ == '__main__':
    main()