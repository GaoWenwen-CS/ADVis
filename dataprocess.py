import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


attribute = ['性别（1女、2男）','民族（1汉族、2维吾尔族、3回族、4哈萨克族、5蒙古族、6乌孜别克族、7达斡尔族、8东乡族）','年龄',
                  '疼痛部位（1胸部、2腹部、3背部、4胸背部、5胸腹、6腹背、7无疼痛）','高血压病史（0无、1有）','吸烟史（0无、1有）',
                  '饮酒史（0无、1有）','收缩压','舒张压','呼吸','心率','身高','体重','BMI','钾','钠','氯','二氧化碳结合力','钙',
                  '镁','磷','尿素','肌酐','尿酸','葡萄糖','甘油三脂','总胆固醇','高密度脂蛋白','低密度脂蛋白','总胆红素','直接胆红素',
                  '非结合胆红素','总蛋白','白蛋白','球蛋白','门冬氨酸氨基转移酶','丙氨酸氨基转移酶','门冬氨酸丙氨酸','乳酸脱氢酶',
                  '谷氨酰转肽酶','碱性磷酸酶','肌酸激酶','渗透压','白细胞','中性粒细胞计数','淋巴细胞计数','单核细胞计数','嗜酸性细胞计数',
                  '嗜碱性细胞计数','红细胞计数','血红蛋白','红细胞压积','平均红细胞体积','红细胞分布宽度','血小板计数', '血小板平均体积',
                  '血小板压积','血小板平均分布宽度','凝血酶原时间过筛实验','PT活动度', '纤维蛋白原','部分凝血活酶时间',
                  '凝血酶时间','APTT正常对照','TT正常对照','D二聚体','纤维蛋白原降解产物FDP', '超敏C反应蛋白']

def dict_data(attribute):
    dic = {}
    for item, value in enumerate(attribute):
        dic[str(item)] = value
    return dic

def normal(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    out = (data - mean) / std
    return out

def normal_1(data):
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)
    out = (data-min)/(max-min)
    return np.round(out*100,1)

def select_feature(input):
    out = []
    dict = dict_data(attribute)
    for i in input:
        out.append(dict[i])
    return out

def load_data(data_path, select_attribute):
    '''
    :param data_path: 原始数据路径
    :param select_attribute:  前端按钮选择的属性序号，序号为字符串，如果没有选择则默认使用全部数据
    数据样例：['0','2','4','8','16']  可以是空列表
    :return: 全部/被选择属性数据，对应标签  data_shape:(718, 属性列), label_shape:(718, 1)
    '''
    all_data = pd.read_excel(data_path, header=0,engine='openpyxl')

    if len(select_attribute) < 1:
        sel_attr = attribute
        data = np.array(all_data.drop(labels='Label', axis=1))
        data = np.array(pd.DataFrame(data, columns=sel_attr))
        # data = normal(data)
    else:
        sel_attr = select_feature(select_attribute)
        data = all_data.drop(labels='Label', axis=1)
        data = np.array(pd.DataFrame(data,columns=sel_attr))
        # data = normal(data)
    label = np.array(all_data['Label']).reshape((-1, 1))
    return data, label, sel_attr

def test_data(test_data_path, select_attribute):

    all_data = pd.read_excel(test_data_path, header=0,engine='openpyxl')

    if len(select_attribute) < 1:
        sel_attr = attribute
        data = np.array(all_data.drop(labels='Label', axis=1))
        data = np.array(pd.DataFrame(data, columns=sel_attr))

    else:
        sel_attr = select_feature(select_attribute)
        data = all_data.drop(labels='Label', axis=1)
        data = np.array(pd.DataFrame(data,columns=sel_attr))

    return data, sel_attr



def normal_all_data(data_path,save_path):
    all_data = pd.read_excel(data_path, header=0)

    data = np.array(all_data.drop(labels='Label', axis=1))
    data = normal_1(data)
    #通过index参数设置行索引，通过columns参数设置列索引
    data = pd.DataFrame(data,columns=attribute)
    data.insert(68, 'Label', np.array(all_data['Label']).ravel())
    data.to_csv(save_path,encoding='utf-8-sig')

def Plot(shape_value):
    attribute_dict = {}
    min = np.min(shape_value, axis=0)
    max = np.max(shape_value, axis=0)
    attention_vector = ((shape_value - min) / ((max - min)+1e-5)).mean(axis=0).flatten()


    for i in range(len(attention_vector)):
        attribute_dict[attribute[i]] = attention_vector[i]
    print(attribute_dict)

    # 绘图
    plt.bar(range(len(attention_vector)), attention_vector)
    plt.title('Attention Mechanism as a function of input dimensions.')
    plt.xlabel("Attribute")
    plt.ylabel("Contribution %")
    plt.show()

# normal_all_data('dataset/data.xlsx',"dataset/normal_data.csv")