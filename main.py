from models import model_predict
from dataprocess import load_data
from sklearn.model_selection import train_test_split
from single_predict import single_pred,test_data


select_attribute = []
data, label, sel_attr = load_data(data_path = 'dataset/data.xlsx', select_attribute=select_attribute)
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=1)

test_data_path = 'dataset/test_data.xlsx'
singel_x_test, _ = test_data(test_data_path, select_attribute)

print('x_train_shape', x_train.shape)
print('x_test_shape:', x_test.shape)

model_predict(x_train, y_train, x_test, y_test, data, select_attribute)
single_pred(x_train,y_train,singel_x_test)