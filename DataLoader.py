import numpy as np
import pandas as pd


def get_data(data_path: str, augment=False) -> dict:
    """
    returns a dictionary with x_data (mfc1, mfc2, mfc3, mfc4) which denotes the flow from the four corners and
    y_data (y1, y2) corresponding to location of leakage in the sheet.
    :param data_path: path to the csv data file
    :param augment: whether to augment the data or not (Use if you are creating a train dataset)
    :return: dictionary containing x_data and y_data
    """
    data = pd.read_csv(data_path)
    data = data.to_numpy()
    x, y = data[:, 2:], data[:, :2]
    if augment:  # flip normal
        augment_x, augment_y = [], []
        augment_x.append(x)
        augment_y.append(y)
        augment = Augmentation()
        flip_x, flip_y = augment.flip(x, y)
        augment_x.append(flip_x)
        augment_y.append(flip_y)
        for angle in [90, 180, 270]:  # rotate and flip in all angles
            rot_x, rot_y = augment.rot(x, y, angle)
            flip_rot_x, flip_rot_y = augment.flip(rot_x, rot_y)
            augment_x.append(rot_x)
            augment_x.append(flip_rot_x)
            augment_y.append(rot_y)
            augment_y.append(flip_rot_y)
        x = np.array(augment_x, dtype=np.float32).reshape(-1, 4)
        y = np.array(augment_y, dtype=np.float32).reshape(-1, 2)
    return {'x_data': x, 'y_data': y}


class Augmentation:
    def __init__(self):
        pass

    @staticmethod
    def flip(x, y):
        flipped_y = np.copy(y)
        flipped_x = np.vstack([x[:, 1], x[:, 0], x[:, 3], x[:, 2]]).T
        flipped_y[:, 0] = -flipped_y[:, 0]
        return flipped_x, flipped_y

    @staticmethod
    def rot(x, y, angle=90):
        assert angle in [90, 180, 270], print("Angle should be in range [90, 180, 270]")
        rad = -angle * np.pi / 180
        rot_matrix = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]], dtype=np.float32)
        rot_y = np.matmul(rot_matrix, y.T).T
        rot_x = np.copy(x)
        if angle == 90:
            rot_x = np.vstack([x[:, 3], x[:, 0], x[:, 1], x[:, 2]]).T
        elif angle == 180:
            rot_x = np.vstack([x[:, 2], x[:, 3], x[:, 0], x[:, 1]]).T
        elif angle == 270:
            rot_x = np.vstack([x[:, 1], x[:, 2], x[:, 3], x[:, 0]]).T
        return rot_x, rot_y


class TestData:
    def __init__(self):
        pass

    def __call__(self):
        x1 = np.linspace(0, 0.5, 20)[:, np.newaxis]
        x3 = 0.5 - x1
        x2 = np.ones_like(x1) * 0.25
        x4 = np.copy(x2)
        test_1 = np.concatenate([x1, x2, x3, x4], axis=1)
        test_2 = np.concatenate([x2, x1, x4, x3], axis=1)
        return {'set1': test_1, "set2": test_2}


# data_directory = "dlr_project_data/"
# # df = pd.read_csv(data_directory + 'leakage_synth_dataset_train_100.csv')
# # df_validation = pd.read_csv(data_directory + 'leakage_synth_dataset_validation_1000.csv')
# # dataset = df.to_numpy()
# # val_dataset = df_validation.to_numpy()
# # X_train, Y_train = dataset[:, 2:], dataset[:, :2]
# # X_validation, Y_validation = val_dataset[:, 2:], val_dataset[:, :2]
# # aug = Augmentation()
# # x_flip, y_flip = aug.rot(X_train, Y_train, 90)
# # print(x_flip)
# get_data_dict = get_data(data_directory + 'leakage_synth_dataset_train_100.csv', augment=True)
# print(get_data_dict)
