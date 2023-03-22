import os

from flask import Flask, jsonify, request
from flask_cors import cross_origin

from models import model_predict
from dataprocess import load_data,test_data
from sklearn.model_selection import train_test_split

import json

from single_predict import single_pred

app = Flask(__name__)

@app.route("/default/result" , methods = ["GET"] )
@cross_origin()
def default_result():
    # 获取前端参数
    select_attributes = []

    # 跑模型
    data, label, sel_attr = load_data(data_path='dataset/data.xlsx', select_attribute=select_attributes)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=1)
    model_predict(x_train, y_train, x_test, y_test, data, select_attributes)

    # 返回结果
    with open("model_result.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    return jsonify(results)

@app.route("/predict/attribute" , methods = ["POST"] )
@cross_origin()
def predict_attribute():
    # 获取前端参数
    request_data = request.get_json()
    select_attributes = request_data.get("select_attributes")
    print(select_attributes)

    # 跑模型
    data, label, sel_attr = load_data(data_path='dataset/data.xlsx', select_attribute=select_attributes)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=1)
    model_predict(x_train, y_train, x_test, y_test, data, select_attributes)

    # 返回结果
    with open("model_result.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    return jsonify(results)


@app.route("/diagnosis" , methods = ["POST"] )
@cross_origin()
def diagnosis():
    # 获取前端参数
    # single_data = request.files.get('test_data')
    # single_data.save(os.path.join('dataset/', 'test_data.xlsx'))  # 保存文件
    file_obj = request.files.get('test_data')
    if file_obj:
        f = open('dataset/test_data.xlsx', 'wb')
    data_t = file_obj.read()
    f.write(data_t)
    f.close()


    # 跑模型
    test_data_path = 'dataset/test_data.xlsx'
    select_attribute = []
    x_train, y_train, sel_attr = load_data(data_path='dataset/data.xlsx', select_attribute=select_attribute)
    x_test, _ = test_data(test_data_path, select_attribute)
    single_pred(x_train, y_train, x_test)

    # 返回结果
    with open("single_result.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    return jsonify(results)


# @app.route('/')
# def hello_world():  # put application's code here
#     return 'Hello World!'


if __name__ == '__main__':
    app.run()
