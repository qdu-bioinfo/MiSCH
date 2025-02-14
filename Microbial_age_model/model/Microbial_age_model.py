# @Author  : wxchen
# @Time    : 2025/2/8 22:41
"""
    Scalp Microbial Age Model
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV, GridSearchCV, ShuffleSplit

def data_preprocessing(Abd_path):
    """
    数据处理
    :param Abd_path: 物种相对丰度表
    :return:
    """
    file_path = Abd_path
    data = pd.read_csv(file_path)
    return data

def feature_importance_ranking(data, target):
    """
    RF进行特种重要性排序
    :param data:数据
    :param target:目标变量real_age
    :return:已排序的特征
    """
    feature_order = []
    X = data.iloc[:,2:]
    y = data[target]
    importance_list = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    # 将数据随机划分10份，每次选取其中9份进行评估特征重要性
    for train_index, _ in kf.split(X):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        feature_importances = model.feature_importances_
        importance_list.append(feature_importances)

    average_importance = np.mean(importance_list,axis=0)
    feature_names = X.columns
    # 将每个物种的物种名称和对应的特征重要性放入字典中
    feature_importances_dict = {feature_name:importance for feature_name,
                                importance in zip(feature_names, average_importance)}
    sorted_features = sorted(feature_importances_dict.items(), key=lambda x: x[1],reverse=True)
    for feature,importance in sorted_features:
        feature_order.append(feature)

    return feature_order

def optimal_features(sorted_features,data,target):
    """
    选择最佳的特征组合
    :param sorted_features: 按照特征重要性排序后的特征列表
    :param data: 丰度表
    :param target: 目标变量
    :return:最佳特征组合
    """
    features = data.iloc[:,2:]
    target = data[target]

    rf_model = RandomForestRegressor(random_state=42)
    kf = KFold(n_splits=10,shuffle=True,random_state=42)
    cv_errors = []

    for i in range(1,len(sorted_features) + 1):
        selected_features = sorted_features[:i]
        cv_score = np.mean(cross_val_score(rf_model,features[selected_features], target, cv=kf,scoring='neg_mean_squared_error'))
        cv_errors.append(-cv_score)

    for i in range(len(cv_errors)):
        plt.scatter(i + 1, cv_errors[i], color='#B3B3B3')
    for i in range(len(cv_errors)-1):
        plt.plot([i + 1, i + 2], [cv_errors[i], cv_errors[i + 1]], color='#B3B3B3')

    min_features_count = np.argmin(cv_errors) + 1
    optimal_features = sorted_features[:min_features_count]
    print(f'Best feature combination:{optimal_features}, Optimal number of features:{len(optimal_features)}')

    plt.xlabel('features count')
    plt.ylabel('CV10_MSE')
    plt.title('Model Performance with Increasing Features')
    plt.show()

    return optimal_features

def model_parameters(data,features,target):
    """
    RandomizedSearchCV大范围内随机搜索寻找最优超参数,或者继续GridSearchCV小范围穷举搜索最佳参数组合
    """
    features_data=data[features]
    target_data = data[target]

    rf_model = RandomForestRegressor(bootstrap=True,random_state=42)
    n_estimators = [int(x) for x in np.linspace(start = 100,stop = 1000,num = 20)]

    min_samples_split = [2, 5, 8]
    min_samples_leaf = [1, 2, 4]
    max_features = [None, 'log2', 'sqrt']
    max_depth = [5, 8, 10]

    random_param_group = {'n_estimators':n_estimators,
                          'min_samples_split':min_samples_split,
                          'min_samples_leaf':min_samples_leaf,
                          'max_features': max_features,
                          'max_depth':max_depth,
                          }
    # RandomizedSearchCV大范围内随机搜索寻找最优超参数
    rf_model = RandomizedSearchCV(rf_model,
                                  param_distributions=random_param_group,
                                  n_iter=200,
                                  scoring='neg_mean_absolute_error',
                                  cv=10,
                                  verbose=2,
                                  n_jobs=-1,
                                  random_state=42)
    rf_model.fit(features_data,target_data)
    print("Best Parameters: ", rf_model.best_params_)
    print("Best Score: ", rf_model.best_score_)

    param_grid = {
        "n_estimators":[int(x) for x in np.linspace(start = 100,stop = 200,num = 6)],
        "min_samples_split":[4,5,6],
        "min_samples_leaf":[1,2,3],
        "max_features": [None, 'log2', 'sqrt'],
        "max_depth":[6,7,8,9]}
    # 小范围穷举搜索最佳参数组合
    # grid = GridSearchCV(rf_model,param_grid,scoring='neg_mean_absolute_error',cv=5,n_jobs=-1)
    # grid.fit(features_data,target_data)
    # print('GridSearchCV策略')
    # print("grid Best Parameters: ", grid.best_params_)
    # print('params类型',type(grid.best_params_))
    # print("grid Best Score: ", grid.best_score_)

    best_params = rf_model.best_params_
    return best_params

def SMA_model(data, features,target,parameters_dict):
    """
    利用健康样本训练出头皮微生物年龄模型
    """
    features_data = data[features]
    target_data = data[target]
    rf_model = RandomForestRegressor(**parameters_dict,bootstrap=True,random_state=42)
    mae_scores = []
    r2_scores = []

    shuffle_split = ShuffleSplit(n_splits=10,test_size=0.2,random_state=42)

    for train_index,test_index in shuffle_split.split(features_data):
        X_train,X_test = features_data.iloc[train_index],features_data.iloc[test_index]
        y_train,y_test = target_data.iloc[train_index],target_data.iloc[test_index]

        rf_model.fit(X_train,y_train)
        y_pred = rf_model.predict(X_test)

        mae_scores.append(mean_absolute_error(y_test,y_pred))
        r2_scores.append(r2_score(y_test,y_pred))

    return rf_model

def SMA_model_pred(SAM_model,scalp_data,features,target):
    """
    利用Scalp Microbial Age Model预测样本的微生物年龄
    """
    features_data = scalp_data[features]
    target_data = scalp_data[target]
    predicted_age = SAM_model.predict(features_data)

    scalp_data['predicted_age'] = predicted_age

    MAE = mean_absolute_error(target_data, predicted_age)
    r2 = r2_score(target_data, predicted_age)
    print('mean_absolute_error',MAE)
    print('r2_score',r2)

    plot_show(scalp_data)

def plot_show(scalp_data):
    """
    微生物年龄曲线图，散点图
    """
    groups = scalp_data['group']
    colors = {'Healthy': 'green', 'AGA': 'red'}

    plt.scatter(scalp_data['real_age'][groups == 'Healthy'], scalp_data['predicted_age'][groups == 'Healthy'],
                color=colors['Healthy'], label='Healthy')
    plt.scatter(scalp_data['real_age'][groups == 'AGA'], scalp_data['predicted_age'][groups == 'AGA'], color=colors['AGA'],
                label='AGA')

    plt.xlabel('Real Age')
    plt.ylabel('Microbiota age')
    plt.title('Microbiota age vs. Real age')
    plt.xticks(np.arange(0, max(scalp_data['real_age']) + 5, 5))
    plt.yticks(np.arange(0, max(scalp_data['predicted_age']) + 5, 5))
    plt.axis('scaled')

    plt.legend()
    plt.show()

def main():

    file_path = r"D:\Github_MiSCH\Microbial_age_model\data\healthy102_abd_composite.csv"
    target = 'real_age'
    data = data_preprocessing(file_path)

    # features_list = feature_importance_ranking(data,target)
    # features = optimal_features(features_list,data,target)
    features = ['g__Paracoccus;', 'g__Micrococcus;', 'g__Rothia;', 'g__Cladosporium;', 'g__Propionibacterium;','g__Acinetobacter;']

    # parameters_dict = model_parameters(data, features, target)
    parameters_dict = {'n_estimators': 905,
                       'min_samples_split': 2,
                       'min_samples_leaf': 1,
                       'max_features': 'sqrt',
                       'max_depth': 10}
    pred_model = SMA_model(data, features, target, parameters_dict)

    pred_data_path = r"D:\Github_MiSCH\Microbial_age_model\data\all_178abd_composite.csv"

    pred_data = data_preprocessing(pred_data_path)
    SMA_model_pred(pred_model,pred_data,features,target)

if __name__ == "__main__":
    main()