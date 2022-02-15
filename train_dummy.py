import tensorflow as tf
from tensorflow import keras
from DataLoader import get_data


model = keras.Sequential()
model.add(keras.layers.Input(shape=[4]))
model.add(keras.layers.Dense(units=32, activation='relu'))
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=2, activation=None))
model.compile(loss='mse', optimizer='Adam')

print(model.summary())
data_directory = "dlr_project_data/"
train_dict = get_data(data_directory + 'leakage_synth_dataset_train_100.csv')
x_train, y_train = train_dict['x_data'], train_dict['y_data']
validation_dict = get_data(data_directory + 'leakage_synth_dataset_validation_1000.csv')
x_validation, y_validation = validation_dict['x_data'], validation_dict['y_data']
history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=20)
print(history)