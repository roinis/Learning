from read_external_datasets import *
from datasets import Dataset
from sklearn.model_selection import KFold
import random


exp = 'general'
arch = 'general_improved'
learning_rate = 0.001
standardise_samples = False
affine_std = 0.1
xlat_range = 2.0
hflip = False
intens_flip = False
intens_scale_range = ''
intens_offset_range = ''
gaussian_noise_std = 0.1
num_epochs = 10
batch_size = 250
seed = 0
log_file = None
device = 0


def read_ds():
    ds_list=[]
    # 1
    print('read flowers_color')
    flowers_colors_X, flowers_colors_Y = read_flowers_colors()
    ds_list.append((flowers_colors_X, flowers_colors_Y, "flowers_color"))

    # 2
    print('read gemstones')
    gemstones_X, gemstones_Y = read_gemstones_dataset()
    ds_list.append((gemstones_X, gemstones_Y, "gemstones"))

    # 3
    print('read brain_mri')
    brain_mri_X, brain_mri_Y = read_brain_mri_dataset()
    ds_list.append((brain_mri_X, brain_mri_Y, "brain_mri"))

    # 4
    print('read covid_xray')
    covid_xray_X, covid_xray_Y = read_covid_xray_dataset()
    ds_list.append((covid_xray_X, covid_xray_Y, "covid_xray"))

    # 5
    print('read hurricane')
    hurricane_X, hurricane_Y = read_hurricane_damage_dataset()
    ds_list.append((hurricane_X, hurricane_Y, "hurricane"))

    # 6
    print('read vegetables')
    vegetables_X, vegetables_Y = read_vegetables_dataset()
    ds_list.append((vegetables_X, vegetables_Y, "vegetables"))

    # 7
    print('read clothes_color')
    clothes_X, clothes_Y = read_clothes_colors_dataset()
    ds_list.append((clothes_X, clothes_Y, "clothes_color"))

    # 8
    print('read rooms')
    rooms_X, rooms_Y = read_clean_messy_room_dataset()
    ds_list.append((rooms_X, rooms_Y, "rooms"))

    # 9
    print('read traffic_recognition')
    traffic_X, traffix_Y = read_traffic_recognition_dataset()
    ds_list.append((traffic_X, traffix_Y, "traffic_recognition"))

    # 10
    print('read sign_language')
    sign_language_X, sign_language_Y = read_sign_language_dataset()
    ds_list.append((sign_language_X, sign_language_Y, "sign_language"))

    # 11
    print('read shapes')
    shapes_X, shapes_Y = read_shapes_dataset()
    ds_list.append((shapes_X, shapes_Y, "shapes"))

    # 12
    print('read brain_tumor')
    brain_tumor_X, brain_tumor_Y = read_brain_tumor_dataset()
    ds_list.append((brain_tumor_X, brain_tumor_Y, "brain_tumor"))

    # 13
    print('read wildfire')
    wildfire_X, wildfire_Y = read_wildfire_dataset()
    ds_list.append((wildfire_X, wildfire_Y, "wildfire"))

    # 14
    print('read tom_and_jerry')
    tom_jerry_X, tom_jerry_Y = read_tom_jerry_dataset()
    ds_list.append((tom_jerry_X, tom_jerry_Y, "tom_and_jerry"))

    # 15
    print('read hand_gustures')
    hand_gustures_X, hand_gustures_Y = read_hand_gestures_dataset()
    ds_list.append((hand_gustures_X, hand_gustures_Y, "hand_gustures"))

    # 16
    print('read bows')
    bows_X, bows_Y = read_bows_dataset()
    ds_list.append((bows_X, bows_Y, "bows"))

    # 17
    print('read iris')
    iris_X, iris_Y = read_iris_dataset()
    ds_list.append((iris_X, iris_Y, "iris"))

    # 18
    print('read vehicles')
    vehicles_X, vehicles_Y = read_vehicles_dataset()
    ds_list.append((vehicles_X, vehicles_Y, "vehicles"))

    # 19
    print('read cellphones')
    cellphones_X, cellphones_Y = read_cellphone_dataset()
    ds_list.append((cellphones_X, cellphones_Y, "cellphones"))

    # 20
    print('read indian_currency')
    indian_currency_X, indian_currency_Y = read_indian_currency()
    ds_list.append((indian_currency_X, indian_currency_Y, "indian_currency"))

    transposed_ds_list = []
    for ds_x,ds_y,ds_name in ds_list:
        ds_x_new = ds_x.transpose((0,3,1,2))
        transposed_ds_list.append(Dataset(ds_x_new,ds_y,ds_name))
    return transposed_ds_list


def kfold_cross_validation(exp, arch, learning_rate,
                           standardise_samples, affine_std, xlat_range, hflip,
                           intens_flip, intens_scale_range, intens_offset_range, gaussian_noise_std,
                           num_epochs, batch_size, seed,
                           log_file, device):

    dses = read_ds()
    for ds in dses:
        cv_outer_source = KFold(n_splits=10, shuffle=True, random_state=1)
        cv_outer_target = KFold(n_splits=10, shuffle=True, random_state=1)
        d_source = random.sample(dses)
        d_target = ds
        counter = 0
        target_kf = list(cv_outer_target.split(d_target.train_X))
        for train_ix, test_ix in cv_outer_source.split(d_source.X):
            d_source.train_X, d_source.test_X = d_source.X[train_ix, :], d_source.X[test_ix, :]
            d_source.train_y, d_source.test_y = d_source.y[train_ix], d_source.y[test_ix]

            d_target.train_X, d_target.test_X = d_target.X[target_kf[counter][0], :], d_target.X[target_kf[counter][1], :]
            d_target.train_y, d_target.test_y = d_target.y[train_ix], d_target.y[test_ix]


            experiment(exp, arch, learning_rate,
                       standardise_samples, affine_std, xlat_range, hflip,
                       intens_flip, intens_scale_range, intens_offset_range, gaussian_noise_std,
                       num_epochs, batch_size, seed,
                       log_file, device, d_source, d_target)
            counter += 1


def experiment(exp, arch, learning_rate,
               standardise_samples, affine_std, xlat_range, hflip,
               intens_flip, intens_scale_range, intens_offset_range, gaussian_noise_std,
               num_epochs, batch_size, seed,
               log_file, device,d_source,d_target):
    import os
    import sys
    import cmdline_helpers

    if log_file == '':
        log_file = 'output_aug_log_{}.txt'.format(exp)
    elif log_file == 'none':
        log_file = None

    if log_file is not None:
        if os.path.exists(log_file):
            print('Output log file {} already exists'.format(log_file))
            return

    intens_scale_range_lower, intens_scale_range_upper, intens_offset_range_lower, intens_offset_range_upper = \
        cmdline_helpers.intens_aug_options(intens_scale_range, intens_offset_range)

    import time
    import math
    import numpy as np
    from batchup import data_source, work_pool
    import data_loaders
    import standardisation
    import network_architectures
    import augmentation
    import torch, torch.cuda
    from torch import nn
    from torch.nn import functional as F

    with torch.cuda.device(device):
        pool = work_pool.WorkerThreadPool(2)

        n_chn = 0

        # Delete the training ground truths as we should not be using them
        del d_target.train_y

        if standardise_samples:
            standardisation.standardise_dataset(d_source)
            standardisation.standardise_dataset(d_target)

        n_classes = d_source.n_classes

        print('Loaded data')

        net_class, expected_shape = network_architectures.get_net_and_shape_for_architecture(arch)

        if expected_shape != d_source.train_X.shape[1:]:
            print('Architecture {} not compatible with experiment {}; it needs samples of shape {}, '
                  'data has samples of shape {}'.format(arch, exp, expected_shape, d_source.train_X.shape[1:]))
            return

        net = net_class(n_classes).cuda()
        params = list(net.parameters())

        optimizer = torch.optim.Adam(params, lr=learning_rate)
        classification_criterion = nn.CrossEntropyLoss()

        print('Built network')

        aug = augmentation.ImageAugmentation(
            hflip, xlat_range, affine_std,
            intens_scale_range_lower=intens_scale_range_lower, intens_scale_range_upper=intens_scale_range_upper,
            intens_offset_range_lower=intens_offset_range_lower, intens_offset_range_upper=intens_offset_range_upper,
            intens_flip=intens_flip, gaussian_noise_std=gaussian_noise_std)

        def augment(X_sup, y_sup):
            X_sup = aug.augment(X_sup)
            return [X_sup, y_sup]

        def f_train(X_sup, y_sup):
            X_sup = torch.autograd.Variable(torch.from_numpy(X_sup).cuda())
            y_sup = torch.autograd.Variable(torch.from_numpy(y_sup).long().cuda())

            optimizer.zero_grad()
            net.train(mode=True)

            sup_logits_out = net(X_sup)

            # Supervised classification loss
            clf_loss = classification_criterion(sup_logits_out, y_sup)

            loss_expr = clf_loss

            loss_expr.backward()
            optimizer.step()

            n_samples = X_sup.size()[0]

            return float(clf_loss.data.cpu().numpy()) * n_samples

        print('Compiled training function')

        def f_pred_src(X_sup):
            X_var = torch.autograd.Variable(torch.from_numpy(X_sup).cuda())
            net.train(mode=False)
            return F.softmax(net(X_var)).data.cpu().numpy()

        def f_pred_tgt(X_sup):
            X_var = torch.autograd.Variable(torch.from_numpy(X_sup).cuda())
            net.train(mode=False)
            return F.softmax(net(X_var)).data.cpu().numpy()

        def f_eval_src(X_sup, y_sup):
            y_pred_prob = f_pred_src(X_sup)
            y_pred = np.argmax(y_pred_prob, axis=1)
            return float((y_pred != y_sup).sum())

        def f_eval_tgt(X_sup, y_sup):
            y_pred_prob = f_pred_tgt(X_sup)
            y_pred = np.argmax(y_pred_prob, axis=1)
            return float((y_pred != y_sup).sum())

        print('Compiled evaluation function')

        # Setup output
        def log(text):
            print(text)
            if log_file is not None:
                with open(log_file, 'a') as f:
                    f.write(text + '\n')
                    f.flush()
                    f.close()

        cmdline_helpers.ensure_containing_dir_exists(log_file)

        # Report setttings
        log('sys.argv={}'.format(sys.argv))

        # Report dataset size
        log('Dataset:')
        log('SOURCE Train: X.shape={}, y.shape={}'.format(d_source.train_X.shape, d_source.train_y.shape))
        log('SOURCE Test: X.shape={}, y.shape={}'.format(d_source.test_X.shape, d_source.test_y.shape))
        log('TARGET Train: X.shape={}'.format(d_target.train_X.shape))
        log('TARGET Test: X.shape={}, y.shape={}'.format(d_target.test_X.shape, d_target.test_y.shape))

        print('Training...')
        train_ds = data_source.ArrayDataSource([d_source.train_X, d_source.train_y]).map(augment)

        source_test_ds = data_source.ArrayDataSource([d_source.test_X, d_source.test_y])
        target_test_ds = data_source.ArrayDataSource([d_target.test_X, d_target.test_y])

        if seed != 0:
            shuffle_rng = np.random.RandomState(seed)
        else:
            shuffle_rng = np.random

        best_src_test_err = 1.0
        for epoch in range(num_epochs):
            t1 = time.time()

            train_res = train_ds.batch_map_mean(
                f_train, batch_size=batch_size, shuffle=shuffle_rng)

            train_clf_loss = train_res[0]
            src_test_err, = source_test_ds.batch_map_mean(f_eval_src, batch_size=batch_size * 4)
            tgt_test_err, = target_test_ds.batch_map_mean(f_eval_tgt, batch_size=batch_size * 4)

            t2 = time.time()

            if src_test_err < best_src_test_err:
                log('*** Epoch {} took {:.2f}s: TRAIN clf loss={:.6f}; '
                    'SRC TEST ERR={:.3%}, TGT TEST err={:.3%}'.format(
                    epoch, t2 - t1, train_clf_loss, src_test_err, tgt_test_err))
                best_src_test_err = src_test_err
            else:
                log('Epoch {} took {:.2f}s: TRAIN clf loss={:.6f}; '
                    'SRC TEST ERR={:.3%}, TGT TEST err={:.3%}'.format(
                    epoch, t2 - t1, train_clf_loss, src_test_err, tgt_test_err))


if __name__ == '__main__':
    experiment()
