import numpy as np
import tensorflow as tf
from DataLoader import get_data
from Model import build_model, build_enn_model
import keras_tuner as kt
from tensorflow import keras


class MyTuner(kt.tuners.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 256, step=32, default=32)
        kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 50)
        super(MyTuner, self).run_trial(trial, *args, **kwargs)


data_directory = "dlr_project_data/"
train_dict = get_data(data_directory + 'leakage_synth_dataset_train_100.csv', augment=False)
x_train, y_train = train_dict['x_data'], train_dict['y_data']
validation_dict = get_data(data_directory + 'leakage_synth_dataset_validation_1000.csv')
x_validation, y_validation = validation_dict['x_data'], validation_dict['y_data']
train_dataset = tf.data.Dataset.from_tensor_slices((x_train.astype(np.float32), y_train.astype(np.float32)))
batch_size = 2
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((x_validation.astype(np.float32), y_validation.astype(np.float32)))
val_dataset = val_dataset.batch(batch_size)
test_dict = get_data(data_directory + 'leakage_synth_dataset_test_1000.csv')
x_test, y_test = test_dict['x_data'], test_dict['y_data']
# tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=100, executions_per_trial=2, directory="",
#                         project_name="DLR_ArchitectureSearch10000Aug")
tuner = kt.RandomSearch(build_enn_model, objective='val_loss', max_trials=100, executions_per_trial=1, directory="",
                        project_name="DLR_ArchitectureSearch100ENN")
print('searching done')
tuner.search(train_dataset, epochs=100, validation_data=val_dataset,
             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5,
                                                      restore_best_weights=True),
                        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1,
                                                          mode='min', min_lr=1e-5)])
best_model = tuner.get_best_models(num_models=1)[0]
best_model.build(input_shape=[1, 4])
print(best_model.summary())
print(tuner.get_best_hyperparameters())
best_model.evaluate(x=x_test, y=y_test)
pred_data = x_test[2, :][np.newaxis, ...]
y_pred = best_model.predict(pred_data)
y_true = y_test[2, :]
print(f"y_true: {y_true}, y_pred: {y_pred}")
print('over')
