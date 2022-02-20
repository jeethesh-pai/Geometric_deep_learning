import numpy as np
import tensorflow as tf
import keras_tuner as kt
from DataLoader import get_data, Augmentation
from tensorflow import keras
from EquivariantModel import EquiVariantModel


tf.debugging.experimental.enable_dump_debug_info("Tensorboard/tfdbg2_logdir", tensor_debug_mode="FULL_HEALTH",
                                                 circular_buffer_size=-1)
data_directory = "dlr_project_data/"
train_dict = get_data(data_directory + 'leakage_synth_dataset_train_100.csv', augment=False)
x_train, y_train = train_dict['x_data'], train_dict['y_data']
validation_dict = get_data(data_directory + 'leakage_synth_dataset_validation_1000.csv')
batch_size = 32
x_validation, y_validation = validation_dict['x_data'], validation_dict['y_data']
train_dataset = tf.data.Dataset.from_tensor_slices((x_train.astype(np.float32), y_train.astype(np.float32)))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((x_validation.astype(np.float32), y_validation.astype(np.float32)))
val_dataset = val_dataset.batch(batch_size)
test_dict = get_data(data_directory + 'leakage_synth_dataset_test_1000.csv', augment=False)
x_test, y_test = test_dict['x_data'], test_dict['y_data']
eq_test_x = x_test[0:8000:1000, :]
eq_test_y = y_test[0:8000:1000, :]
my_x = np.array([[1, 2, 3, 4],
                 [4, 1, 2, 3],
                 [3, 4, 1, 2],
                 [2, 3, 4, 1]], dtype=np.float64)
model = EquiVariantModel(num_layers=1, activation='relu')
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mse')
model.build(input_shape=[1, 4])
print(model.trainable_weights)
print(model.summary())
history = model.fit(train_dataset, validation_data=val_dataset, epochs=100,
                    callbacks=[tf.keras.callbacks.TensorBoard(log_dir="Tensorboard/", write_graph=True),
                               keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1,
                                                                 mode='min', min_lr=1e-5)])
y_pred = model.predict(eq_test_x)
my_pred = model.predict(my_x)
print(model.trainable_weights)
check = np.concatenate([eq_test_x, y_pred, eq_test_y], axis=1)
np.set_printoptions(linewidth=150)
print("X_pred, Y_pred, y_actual: \n", check)
print("my_pred: \n", my_pred)






