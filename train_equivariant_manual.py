import math
import csv
import numpy as np
import tensorflow as tf
from DataLoader import get_data
from EquivariantModel import EquiVariantModel
from tqdm import tqdm
from colorama import Fore

#  prepare data
data_directory = "dlr_project_data/"
train_dict = get_data(data_directory + 'leakage_synth_dataset_train_10000.csv', augment=False)
x_train, y_train = train_dict['x_data'], train_dict['y_data']
validation_dict = get_data(data_directory + 'leakage_synth_dataset_validation_1000.csv')
x_validation, y_validation = validation_dict['x_data'], validation_dict['y_data']
test_dict = get_data(data_directory + 'leakage_synth_dataset_test_1000.csv', augment=False)
x_test, y_test = test_dict['x_data'].astype(np.float32), test_dict['y_data'].astype(np.float32)

#  load hyper parameters for checking
layer_variables = [i for i in range(1, 10)]  # number of layers
activation = ['relu', 'sigmoid', 'tanh']  # types of activation
batches = [2, 4, 8, 16, 32, 64]  # no. of batches
start_lr = [0.01, 0.001, 0.0001]  # start learning rate to be used
global_val_loss = {}
for layer in layer_variables:
    for act in activation:
        for lr in start_lr:
            for batch_size in batches:
                batch_lr = lr
                print(f'Model in check: Num Layer: {layer}, Activation: {act}, LR: {lr}, Batch_size: {batch_size}')
                model_name = f"M{layer}_{act}_{lr}_{batch_size}"
                model = EquiVariantModel(num_layers=layer, activation=act)
                opt = tf.keras.optimizers.Adam(learning_rate=lr)

                # use tf.data to load data set efficiently
                train_dataset = tf.data.Dataset.from_tensor_slices((x_train.astype(np.float32),
                                                                    y_train.astype(np.float32)))
                train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
                val_dataset = tf.data.Dataset.from_tensor_slices((x_validation.astype(np.float32),
                                                                  y_validation.astype(np.float32)))
                val_dataset = val_dataset.shuffle(buffer_size=1024).batch(32)
                epochs = 200  # actual metric would be controlled by early stopping callback
                train_bar = tqdm(range(epochs), colour=Fore.GREEN)
                temp_val_loss = []
                val_loss = 0.0
                cooldown = 0
                for epoch in train_bar:
                    # train_bar.set_description(f"Training Epoch -- {epoch + 1} / {epochs}")
                    batch_loss = []
                    for (x_batch, y_batch) in train_dataset:
                        with tf.GradientTape() as tape:
                            y_pred = model(x_batch, training=True)
                            loss = tf.reduce_mean((y_batch - y_pred) ** 2)
                        batch_loss.append(loss.numpy())
                        gradients = tape.gradient(loss, model.trainable_weights)
                        opt.apply_gradients(zip(gradients, model.trainable_weights))
                        train_bar.set_description(f"Training Epoch -- {epoch + 1} / {epochs} - "
                                                  f"Loss: {np.mean(batch_loss)}, val_loss: {np.mean(temp_val_loss)}")
                    batch_val_loss = []
                    train_bar.set_description(f"Validation Epoch -- {epoch + 1} / {epochs}")
                    for (x_batch, y_batch) in val_dataset:
                        y_pred = model(x_batch, training=False)
                        val_loss = tf.reduce_mean((y_batch - y_pred) ** 2)
                        batch_val_loss.append(val_loss.numpy())
                    train_bar.set_description(f"Validation Epoch -- {epoch + 1} / {epochs} - "
                                              f"Loss: {np.mean(batch_loss)}, val_loss: {np.mean(batch_val_loss)}")
                    temp_val_loss.append(np.mean(batch_val_loss))
                    if len(temp_val_loss) > 5:
                        if not math.isclose(temp_val_loss[-2], temp_val_loss[-1], rel_tol=1e-4):  # Reduce LR on plateau
                            cooldown += 1
                    if math.isclose(batch_lr, 1e-5, rel_tol=1e-4) and cooldown > 5:  # Early stopping
                        train_bar.set_description(f"Validation Epoch -- {epoch + 1} / {epochs} - "
                                                  f"Loss: {np.mean(batch_loss)}, val_loss: {np.mean(batch_val_loss)} "
                                                  f"Stopping \n")
                        cooldown = 0
                        break
                    if cooldown > 5:
                        new_lr = batch_lr * 0.1
                        opt.lr = new_lr
                        train_bar.set_description(f"Validation Epoch -- {epoch + 1} / {epochs} - "
                                                  f"Loss: {np.mean(batch_loss)}, "
                                                  f"val_loss: {np.mean(batch_val_loss)},"
                                                  f"Reducing LR {batch_lr}-> {new_lr} \n")
                        batch_lr = new_lr
                        cooldown = 0
                print("\n")
                global_val_loss[model_name] = np.mean(batch_val_loss)
print(global_val_loss)
with open('loss10000.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in global_val_loss.items():
        writer.writerow([key, value])
print("training complete .... ")
