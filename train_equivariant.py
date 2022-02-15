import tensorflow as tf
import keras_tuner as kt
from DataLoader import get_data
from tensorflow import keras
from EquivariantModel import EquiVariantModel


data_directory = "dlr_project_data/"
train_dict = get_data(data_directory + 'leakage_synth_dataset_train_10000.csv', augment=True)
x_train, y_train = train_dict['x_data'], train_dict['y_data']
validation_dict = get_data(data_directory + 'leakage_synth_dataset_validation_1000.csv')
x_validation, y_validation = validation_dict['x_data'], validation_dict['y_data']
test_dict = get_data(data_directory + 'leakage_synth_dataset_test_1000.csv')
x_test, y_test = test_dict['x_data'], test_dict['y_data']
model = EquiVariantModel(num_layers=2, activation='relu')
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mse')
model.build(input_shape=[1, 4])
print(model.summary())
for trainable in tf.compat.v1.trainable_variables():
    print(trainable)
history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=20,
                    callbacks=[tf.keras.callbacks.TensorBoard(log_dir="Tensorboard/", write_graph=True)])
print(history)



