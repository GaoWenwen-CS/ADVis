import numpy as np
from dataprocess import load_data,test_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import json


# test_data_path = 'dataset/test_data.xlsx'
# select_attribute = []
# x_train, y_train, sel_attr = load_data(data_path = 'dataset/data.xlsx', select_attribute=select_attribute)
# x_test, _ = test_data(test_data_path, select_attribute)

def dict_test_id(x_test,pred):
    test_dict = {}
    for i in range(x_test.shape[0]):
        test_dict[i] = pred[i]
    return test_dict

def single_pred(x_train,y_train,x_test):

    xgb = XGBClassifier().fit(x_train, y_train, eval_metric="logloss", verbose=False)
    Xgb_pred =np.round(np.array(xgb.predict_proba(x_test)[:, 1].tolist()),2)
    Xgb_pred = dict_test_id(x_test, Xgb_pred)

    Lgbm = LGBMClassifier( num_leaves=31, learning_rate=0.05, n_estimators=20).fit(x_train, y_train)
    Lgbm_pred = np.round(np.array(Lgbm.predict_proba(x_test)[:, 1].tolist()), 2)
    Lgbm_pred = dict_test_id(x_test, Lgbm_pred)

    Lda = LDA().fit(x_train, y_train)
    Lda_pred = np.round(np.array(Lda.predict_proba(x_test)[:, 1].tolist()),2)
    Lda_pred = dict_test_id(x_test, Lda_pred)

    '''
    #暂时不用
    MLP_pred, _ = MLPTrain(x_train, y_train, x_test, Trans=False)
    MLP_pred = np.round(np.array(MLP_pred.tolist()),2)
    MLP_pred = dict_test_id(x_test, MLP_pred)
    '''
    Rf = RandomForestClassifier(criterion='gini',max_features='auto',min_samples_leaf=1,min_samples_split=2,n_estimators=600).fit(x_train , np.array(y_train).ravel())
    Rf_pred = np.round(np.array(Rf.predict_proba(x_test)[:, 1].tolist()),2)
    Rf_pred = dict_test_id(x_test, Rf_pred)

    Bays = naive_bayes.GaussianNB().fit(x_train, y_train)
    Bays_pred = np.round(np.array(Bays.predict_proba(x_test)[:, 1].tolist()),2)
    Bays_pred = dict_test_id(x_test, Bays_pred)

    result_dict = dict({'XGB': Xgb_pred, 'LGBM': Lgbm_pred, 'LDA': Lda_pred, 'RF': Rf_pred,'Bayes': Bays_pred})

    with open('single_result.json', 'w', encoding='utf-8') as json_file:
        json.dump(result_dict, json_file, ensure_ascii=False)

    return

# single_pred(x_train,y_train,x_test)