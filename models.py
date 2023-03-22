from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import numpy as np
from metrics import metrics_
import json
import shap
from dataprocess import attribute,dict_data
import pandas as pd



def Contribution(value,select_attribute):

    attribute_dict = {}
    min = np.min(value, axis=0)
    max = np.max(value, axis=0)
    attention_vector = ((value - min) / ((max - min)+1e-5)).mean(axis=0).flatten()

    if len(select_attribute) < 1:
        attribute_dict = dict_data(attribute)
        for item, value in enumerate(attribute_dict):
            attribute_dict[str(item)] = format(attention_vector[item], '.4f')
    else:
        for item,value in enumerate(select_attribute):
            attribute_dict[value] = format(attention_vector[item], '.4f')

    return attribute_dict


def id_test_data(data,x_test):
    test_id = []
    for i in x_test:
        for item,value in enumerate(data):
            if all(i==value):
                test_id.append(item)
    return test_id


def patient_level(pred, label, data, x_test):
    TP, TN, FP, FN= 0, 0, 0, 0
    test_id = id_test_data(data, x_test)
    patient_dict = {}
    for i in range(len(label)):
        if pred[i]==label[i]==0:
            TP += 1
            patient_dict[test_id[i]] = 'tp'
        elif pred[i]==label[i]==1:
            TN += 1
            patient_dict[test_id[i]] = 'tn'
        elif pred[i] != label[i] and label[i]==0:
            FP += 1
            patient_dict[test_id[i]] = 'fp'
        else:
            FN += 1
            patient_dict[test_id[i]] = 'fn'

    return patient_dict


'''
暂时不用
class MLPClass(nn.Module):
    def __init__(self, input_dim):
        super(MLPClass,self).__init__()
        self.input_dim = input_dim

        self.liner1 = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.Sigmoid()
        )

        self.liner2 = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(inplace=True)
        )

        self.liner3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True)
        )

        self.liner4 = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_probs = self.liner1(x)
        out = self.liner2(attention_probs * x)
        out = self.liner3(out)
        out = self.liner4(out)
        return out, attention_probs

def MLPTrain(x_train, y_train, x_test, Trans = True):
    model = MLPClass(x_train.shape[1])
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    for _ in tqdm(range(400)):
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        outputs, _ = model(x_train)
        loss = nn.BCELoss()(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred, attention_probs = model(x_test)

    pred = pred.detach().numpy().ravel()
    attention_probs = attention_probs.detach().numpy()
    if Trans:
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
    else:
        pred = pred

    return pred, attention_probs
'''

def model_predict(x_train, y_train, x_test, y_test, data, select_attribute):
    metrics = ['acc','ppv','tpr','tnr','cm']
    #-----------------------------XGBClassifier-------------------------------------
    xgb = XGBClassifier().fit(x_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=[(x_test, y_test)], verbose=False)
    Xgb_result = metrics_(xgb.predict(x_test), y_test, metrics)
    Xgb_patient_res = patient_level(xgb.predict(x_test), np.array(y_test).ravel(), data, x_test)
    Xgb_result.append(Xgb_patient_res)
    #贡献度计算
    shap_values = shap.TreeExplainer(xgb).shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
    contribution = Contribution(shap_values,select_attribute)
    Xgb_result.append(contribution)
    # -----------------------------XGBClassifier-------------------------------------

    # -----------------------------SVM----------------------------------------------
    Lgbm = LGBMClassifier( num_leaves=31, learning_rate=0.05, n_estimators=20).fit(x_train, y_train)
    Lgbm_result = metrics_(Lgbm.predict(x_test),y_test, metrics)
    Lgbm_patient_res = patient_level(Lgbm.predict(x_test), np.array(y_test).ravel(), data, x_test)
    Lgbm_result.append(Lgbm_patient_res)
    # 贡献度计算
    shap_values = shap.TreeExplainer(Lgbm).shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
    contribution = Contribution(shap_values, select_attribute)
    Lgbm_result.append(contribution)
    # -----------------------------SVM----------------------------------------------

    # -----------------------------LDA----------------------------------------------
    Lda = LDA().fit(x_train, y_train)
    Lda_result = metrics_(Lda.predict(x_test), y_test, metrics)
    Lda_patient_res = patient_level(Lda.predict(x_test), np.array(y_test).ravel(), data, x_test)
    Lda_result.append(Lda_patient_res)
    # 贡献度计算
    shap_values = shap.LinearExplainer(Lda,x_test).shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
    contribution = Contribution(shap_values, select_attribute)
    Lda_result.append(contribution)
    # -----------------------------LDA----------------------------------------------

    # -----------------------------MLP----------------------------------------------
    '''
    #暂时不用
    pred, attention = MLPTrain(x_train, y_train, x_test)
    Mlp_result = metrics_(pred, y_test, metrics)
    Mlp_patient_res = patient_level(pred, np.array(y_test).ravel(), data, x_test)
    Mlp_result.append(Mlp_patient_res)
    # #贡献度计算
    contribution = Contribution(attention, select_attribute)
    Mlp_result.append(contribution)
    '''
    # -----------------------------MLP----------------------------------------------

    # -----------------------------RF----------------------------------------------
    Rf = RandomForestClassifier(criterion='gini',max_features='auto',min_samples_leaf=1,min_samples_split=2,n_estimators=600).fit(x_train , np.array(y_train).ravel())
    Rf_result = metrics_(Rf.predict(x_test), y_test, metrics)
    Rf_patient_res = patient_level(Rf.predict(x_test), np.array(y_test).ravel(), data, x_test)
    Rf_result.append(Rf_patient_res)
    # 贡献度计算
    shap_values = shap.TreeExplainer(Rf).shap_values(x_train)
    contribution = Contribution(shap_values, select_attribute)
    Rf_result.append(contribution)
    # -----------------------------RF----------------------------------------------

    # -----------------------------Bays----------------------------------------------
    Bays = naive_bayes.GaussianNB().fit(x_train, y_train)
    Bays_result = metrics_(Bays.predict(x_test), y_test, metrics)
    Bays_patient_res = patient_level(Bays.predict(x_test), np.array(y_test).ravel(), data, x_test)
    Bays_result.append(Bays_patient_res)
    #暂无贡献值
    Bays_result.append(contribution)
    # -----------------------------Bays----------------------------------------------

    result_dict = dict({'XGB':Xgb_result,'LGBM':Lgbm_result,'LDA':Lda_result,'RF':Rf_result,'Bayes':Bays_result})

    with open('model_result.json', 'w', encoding='utf-8') as json_file:
        json.dump(result_dict, json_file, ensure_ascii=False)

    # pd.DataFrame(result_dict).to_csv('model_result.csv')


