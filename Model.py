from tensorflow import keras
from EquivariantModel import EquiVariantModel


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(hp.Choice('units', [8, 16, 32, 128, 256, 512]),
                                 hp.Choice("activation", ['relu', 'tanh', 'sigmoid']), input_shape=[4]))
    for i in range(hp.Int("num_layers", min_value=1, max_value=4, step=1, default=1)):
        model.add(keras.layers.Dense(hp.Choice(f'units_{i}', [8, 16, 32, 128, 256, 512]),
                                     hp.Choice("activation", ['relu', 'tanh', 'sigmoid'])))
        if hp.Boolean("dropout"):
            model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Dense(2, activation=None))
    lr = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model


def build_enn_model(hp):
    model = EquiVariantModel(hp.Int("num_layers", min_value=1, max_value=10, step=1, default=1),
                             hp.Choice("activation", ['relu', 'tanh', 'sigmoid']))
    lr = hp.Float('lr', min_value=1e-5, max_value=1e-2, sampling='log', default=1e-3)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model


