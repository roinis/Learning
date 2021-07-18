import pandas as pd
# import talos as ta
import random
import data_loaders
import numpy as np
import time
from read_external_datasets import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import KFold


def run_resnet50():
    ds_obj = []
    # ds_obj.append((data_loaders.load_svhn(),"SVHN")) #1 - SVHN
    # ds_obj.append((data_loaders.load_cifar10(),"CIFAR10")) #2 - CIFAR-10
    # ds_obj.append((data_loaders.load_stl(),"STL")) #3 - STL
    # ds_obj.append(data_loaders.load_gtsrb()) #4 - GTSRB
    # ds_obj.append((data_loaders.load_usps(),"USPS")) #5 - USPS
    # ds_obj.append((data_loaders.load_fashion_mnist(),"FMNIST")) #6 - F-MNIST
    # ds_obj.append((data_loaders.load_mnist(),"MNIST")) #7 - MNIST
    # ds_obj.append(data_loaders.load_syn_digits()) #8 - SYN_DIGITS
    # ds_obj.append(data_loaders.load_syn_signs()) #9 - SYN_SIGNS

    ds_list = []

    for ds in ds_obj:
        X_list = []
        X = np.concatenate((ds[0].train_X, ds[0].test_X), axis=0)
        for img in X:
            X_list.append((img.transpose((-1, 1, 0))))
        X = np.asarray(X_list)
        Y = np.concatenate((ds[0].train_y, ds[0].test_y), axis=0)
        name = ds[1]
        ds_list.append((X, Y, name))

    # 1
    print('read flowers_color')
    flowers_colors_X, flowers_colors_Y = read_flowers_colors()
    ds_list.append((flowers_colors_X, flowers_colors_Y, "flowers_color"))

    # 2
    print('read gemstones')
    gemstones_X , gemstones_Y = read_gemstones_dataset()
    ds_list.append((gemstones_X, gemstones_Y, "gemstones"))

    # 3
    print('read brain_mri')
    brain_mri_X , brain_mri_Y = read_brain_mri_dataset()
    ds_list.append((brain_mri_X,brain_mri_Y,"brain_mri"))

    # 4
    print('read covid_xray')
    covid_xray_X , covid_xray_Y = read_covid_xray_dataset()
    ds_list.append((covid_xray_X,covid_xray_Y,"covid_xray"))

    # 5
    print('read hurricane')
    hurricane_X , hurricane_Y = read_hurricane_damage_dataset()
    ds_list.append((hurricane_X,hurricane_Y,"hurricane"))

    # 6
    print('read vegetables')
    vegetables_X , vegetables_Y = read_vegetables_dataset()
    ds_list.append((vegetables_X,vegetables_Y,"vegetables"))

    # 7
    print('read clothes_color')
    clothes_X , clothes_Y = read_clothes_colors_dataset()
    ds_list.append((clothes_X,clothes_Y,"clothes_color"))

    # 8
    print('read rooms')
    rooms_X , rooms_Y = read_clean_messy_room_dataset()
    ds_list.append((rooms_X,rooms_Y,"rooms"))

    # 9
    print('read traffic_recognition')
    traffic_X , traffix_Y = read_traffic_recognition_dataset()
    ds_list.append((traffic_X,traffix_Y,"traffic_recognition"))

    # 10
    print('read sign_language')
    sign_language_X , sign_language_Y = read_sign_language_dataset()
    ds_list.append((sign_language_X,sign_language_Y,"sign_language"))

    #11
    print('read shapes')
    shapes_X , shapes_Y = read_shapes_dataset()
    ds_list.append((shapes_X,shapes_Y,"shapes"))

    #12
    print('read brain_tumor')
    brain_tumor_X , brain_tumor_Y = read_brain_tumor_dataset()
    ds_list.append((brain_tumor_X,brain_tumor_Y,"brain_tumor"))

    #13
    print('read wildfire')
    wildfire_X , wildfire_Y = read_wildfire_dataset()
    ds_list.append((wildfire_X,wildfire_Y,"wildfire"))

    #14
    print('read tom_and_jerry')
    tom_jerry_X , tom_jerry_Y = read_tom_jerry_dataset()
    ds_list.append((tom_jerry_X,tom_jerry_Y,"tom_and_jerry"))

    # 15
    print('read hand_gustures')
    hand_gustures_X , hand_gustures_Y = read_hand_gestures_dataset()
    ds_list.append((hand_gustures_X,hand_gustures_Y,"hand_gustures"))

    #16
    print('read bows')
    bows_X , bows_Y = read_bows_dataset()
    ds_list.append((bows_X,bows_Y,"bows"))

    #17
    print('read iris')
    iris_X , iris_Y = read_iris_dataset()
    ds_list.append((iris_X,iris_Y,"iris"))

    #18
    print('read vehicles')
    vehicles_X , vehicles_Y = read_vehicles_dataset()
    ds_list.append((vehicles_X,vehicles_Y,"vehicles"))

    #19
    print('read cellphones')
    cellphones_X , cellphones_Y = read_cellphone_dataset()
    ds_list.append((cellphones_X,cellphones_Y,"cellphones"))

    #20
    print('read indian_currency')
    indian_currency_X , indian_currency_Y = read_indian_currency()
    ds_list.append((indian_currency_X,indian_currency_Y,"indian_currency"))

    ds_names = []
    algo_names = []
    cross_validation = []
    accuracy, train_accuracy = [], []
    TPR, train_TPR = [], []
    FPR, train_FPR = [], []
    precision, train_precision = [], []
    auc, train_auc = [], []
    prcurve, train_prcurve = [], []
    training_times = []
    inference_times = []
    p = get_params()
    algo_name = 'RESNET50'
    for ds in ds_list:
        name = ds[2]
        print(f'Train DS: {name}...')
        X = ds[0]
        Y = ds[1]
        num_of_classes = np.unique(Y).size
        Y = to_categorical(Y)

        kf = KFold(n_splits=10, shuffle=True)
        cf_index = 1
        model = get_model(32,num_of_classes)
        for train_index, val_index in kf.split(X, Y):
            # scan_object = ta.Scan(x=X[train_index, :],
            #             y=Y[train_index],
            #             model=model,
            #             grid_downsample=0.01,
            #             params=p,
            #             dataset_name=name,
            #             experiment_no='1')
            #
            # predict = ta.Predict(scan_object)

            start_training_time = time.time()
            history = model.fit(X[train_index, :], Y[train_index],
                                batch_size=250,
                                epochs=10)
            training_time = time.time() - start_training_time

            # True Positive = Sensitivity
            # False Positive = (FP)/(FP+TN)
            ds_names.append(name)
            algo_names.append(algo_name)
            cross_validation.append(cf_index)
            train_accuracy.append(history.history['accuracy'][-1])
            tpr = history.history['TP'][-1] / (history.history['TP'][-1] + history.history['FN'][-1])
            train_TPR.append(tpr)
            fpr = history.history['FP'][-1] / (history.history['FP'][-1] + history.history['TN'][-1])
            train_FPR.append(fpr)
            train_precision.append(history.history['Precision'][-1])
            train_auc.append(history.history['AUC'][-1])
            train_prcurve.append(history.history['PR-Curve'][-1])
            training_times.append(training_time)

            start_inference_time = time.time()
            scores = model.evaluate(X[val_index, :], Y[val_index], verbose=0)
            inference_time = time.time() - start_inference_time
            inference_times.append(inference_time)

            accuracy.append(scores[1])
            tpr = scores[6] / (scores[6] + scores[9])
            TPR.append(tpr)
            fpr = scores[8] / (scores[8] + scores[7])
            FPR.append(fpr)
            precision.append(scores[3])
            auc.append(scores[4])
            prcurve.append(scores[5])

            cf_index += 1

    headers = ['Dataset Name', 'Algorithm Name', 'Cross Validation [1-10]',
               'Hyper-Parameters Values', 'Accuracy', 'TPR',
               'FPR', 'Precision', 'AUC', 'PR-Curve', 'Training Time', 'Inference Time']

    cols = [ds_names, algo_names, cross_validation, [''] * len(algo_names), train_accuracy, train_TPR, train_FPR,
            train_precision, train_auc,
            train_prcurve, training_times, inference_times]

    pd.DataFrame({name: data for name, data in zip(headers, cols)}, columns=headers).to_csv('training_res.csv',
                                                                                                index=False)


    cols = [ds_names, algo_names, cross_validation, [''] * len(algo_names), accuracy, TPR, FPR, precision, auc,
            prcurve, training_times, inference_times]
    try:
        pd.DataFrame({name: data for name, data in zip(headers, cols)}, columns=headers).to_csv('res.csv', index=False)
    except Exception:
        with open('res_txt.txt','w') as f:
            f.write(cols)
    print()


def get_model(IMG_SIZE,num_of_classes):
    metrics = ["accuracy", tensorflow.keras.metrics.SensitivityAtSpecificity(0.5, name='FPR_TPR'),
               tensorflow.keras.metrics.Precision(name='Precision'),
               tensorflow.keras.metrics.AUC(name='AUC'), tensorflow.keras.metrics.AUC(name="PR-Curve", curve='PR'),
               tensorflow.keras.metrics.TruePositives(name='TP'),
               tensorflow.keras.metrics.TrueNegatives(name='TN'), tensorflow.keras.metrics.FalsePositives(name='FP'),
               tensorflow.keras.metrics.FalseNegatives(name='FN')]

    # metrics = ["accuracy",tensorflow.keras.metrics.AUC()]
    model = Sequential()

    keras_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    for layer in keras_model.layers:
        layer.trainable = False

    model.add(keras_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(500, activation='selu'))
    model.add(Dropout(.4))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='selu'))
    model.add(Dropout(.25))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_of_classes, activation='sigmoid'))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=metrics)

    return model



if __name__ == '__main__':
    run_resnet50()
