from DataLoader import TestData, get_data
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


data_directory = "dlr_project_data/"
Test = TestData(spacing=50)
test_data = Test()
x_test1 = test_data['set1']
x_test2 = test_data['set2']
model = load_model("Models/BestModel_data1000.hdf5")
model.compile(loss='mse', optimizer='adam')
print(model.summary())
test_dict = get_data(data_directory + 'leakage_synth_dataset_test_1000.csv')
x_test, y_test = test_dict['x_data'], test_dict['y_data']
pred_data = x_test[2, :][np.newaxis, ...]
y_pred = model.predict(pred_data)
y_true = y_test[2, :]
print(f"Model Check - y_true: {y_true}, y_pred: {y_pred}")
test_pred1 = model.predict(x_test1)
test_pred2 = model.predict(x_test2)
plt.plot(test_pred1[:, 0], test_pred1[:, 1])
plt.plot(test_pred2[:, 0], test_pred2[:, 1])
plt.title("Best Model of Data - 1000 ")
plt.xlabel("y1_pred")
plt.ylabel("y2_pred")
plt.savefig("PlotData1000Test.png")
plt.show()
print('over')