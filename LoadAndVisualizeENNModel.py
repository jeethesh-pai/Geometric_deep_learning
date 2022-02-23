import tensorflow as tf
import matplotlib.pyplot as plt
from EquivariantModel import EquiVariantModel
from DataLoader import get_data, TestData
import numpy as np


data_directory = "dlr_project_data/"
model = EquiVariantModel(num_layers=3, activation='tanh')
model.build(input_shape=[1, 4])
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
model.load_weights("Models/ENNModel1000Aug.h5")
print(model.summary())

# Check if the loss values matches the best loss in the loss_*.csv file
validation_dict = get_data(data_directory + 'leakage_synth_dataset_validation_1000.csv')
x_validation, y_validation = validation_dict['x_data'], validation_dict['y_data']
val_dataset = tf.data.Dataset.from_tensor_slices((x_validation.astype(np.float32), y_validation.astype(np.float32)))
val_dataset = val_dataset.batch(32)
test_dict = get_data(data_directory + 'leakage_synth_dataset_test_1000.csv', augment=False)
x_test, y_test = test_dict['x_data'].astype(np.float32), test_dict['y_data'].astype(np.float32)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test.astype(np.float32), y_test.astype(np.float32)))
test_dataset = test_dataset.batch(32)
batch_val_loss = []
for (x_batch, y_batch) in val_dataset:
    y_pred = model(x_batch, training=False)
    batch_val_loss.append(tf.reduce_mean((y_batch - y_pred) ** 2))
print("Validation loss:", np.mean(batch_val_loss))
batch_test_loss = []
for (x_batch, y_batch) in test_dataset:
    y_pred = model(x_batch, training=False)
    batch_test_loss.append(tf.reduce_mean((y_batch - y_pred) ** 2))
print("Test loss:", np.mean(batch_test_loss))

# Load the visualization set for plotting
Test = TestData(spacing=50)
test_data = Test()
x_test1 = test_data['set1']
x_test2 = test_data['set2']
test_pred1 = model(x_test1, training=False)
test_pred2 = model(x_test2, training=False)
plt.plot(test_pred1[:, 0], test_pred1[:, 1])
plt.plot(test_pred2[:, 0], test_pred2[:, 1])
plt.title("Best ENN Model of Data - 1000 Aug")
plt.xlabel("y1_pred")
plt.ylabel("y2_pred")
plt.savefig("Plots/PlotDataENN1000AugTest.png")
plt.show()
print('over')

