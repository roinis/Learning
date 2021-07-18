import data_loaders
import os
from PIL import Image
import pandas as pd
import numpy as np
from image_array_convert import ImageArrayUInt8ToFloat32

INCODE_DS = ['mnist', 'cifar10', 'svhn', 'stl', 'usps']
OUTSOURCE_DS = ['bloodcell']

class Preprocess:
    @staticmethod
    def read_images(images_path):
        images_arr=[]
        for image_path in images_path:
            image = Image.open(image_path).convert('RGB')
            image.load()
            image = image.resize((32, 32), Image.ANTIALIAS)
            pixels = np.asarray(image, dtype="int32")


            images_arr.append(pixels)

        return np.asarray(images_arr)

    @staticmethod
    def load_bloodcell():
        images_path = r'C:\Users\roinis\Documents\Degree\learning\blood\dataset-master\dataset-master\JPEGImages'
        images = Preprocess.read_images(images_path)
        labels_path = r'C:\Users\roinis\Documents\Degree\learning\blood\dataset-master\dataset-master\labels.csv'
        labels_df = pd.read_csv(labels_path)
        labels_df['Category'] = labels_df['Category'].fillna(labels_df['Category'].value_counts().index[0])
        labels_string = {label: index for index, label in enumerate(labels_df['Category'].unique())}
        labels_np = labels_df['Category'].map(labels_string).to_numpy()

        return Dataset(images, labels_np)

    @staticmethod
    def read_ds(ds_name, **kwargs):
        if ds_name in INCODE_DS:
            func = getattr(data_loaders, f"load_{ds_name}")
            return func(**kwargs)

        return getattr(Preprocess,f"load_{ds_name}")()

    @staticmethod
    def read_satellite(images_path):
        # folders =['test','test_another','train_another','validation_another']
        folders = ['train_another']
        damage_folders = [f'{images_path}/{folder_name}/damage' for folder_name in folders]
        no_damage_folders = [f'{images_path}/{folder_name}/no_damage' for folder_name in folders]

        damage_paths = []
        no_damage_paths = []
        for folder in damage_folders:
            damage_paths.extend([os.path.join(os.path.abspath(folder), image_name) for image_name in os.listdir(folder)])

        for folder in no_damage_folders:
            no_damage_paths.extend([os.path.join(os.path.abspath(folder), image_name) for image_name in os.listdir(folder)])

        damage_images, no_damage_images = Preprocess.read_images(damage_paths), Preprocess.read_images(no_damage_paths)
        damage_images_labels,no_damage_images_labels = np.asarray([1]*damage_images.shape[0]), np.asarray([0]*no_damage_images.shape[0])

        return np.concatenate((damage_images,no_damage_images), axis=0), np.concatenate((damage_images_labels,no_damage_images_labels), axis=0)

    @staticmethod
    def read_brain_tumor():
        images_path = r'C:\Users\roinis\Documents\Degree\learning\brain_tumor'
        labels_df = pd.read_csv(f'{images_path}/Brain Tumor.csv')
        labels_df['image_path'] = f'{images_path}/Brain Tumor/Brain Tumor/' + labels_df['Image'] + '.jpg'
        # folders =['test','test_another','train_another','validation_another']
        images = Preprocess.read_images(list(labels_df['image_path']))
        labels = np.asarray(list(labels_df['Class']))
        return images, labels


    @staticmethod
    def read_tom_jerry():
        images_path = r'C:\Users\roinis\Documents\Degree\learning\tom_jerry\train\train'
        angry_path = [os.path.join(os.path.abspath(f'{images_path}/angry'), image_name) for image_name in os.listdir(f'{images_path}/angry')]
        happy_path = [os.path.join(os.path.abspath(f'{images_path}/happy'), image_name) for image_name in
                      os.listdir(f'{images_path}/happy')]
        sad_path = [os.path.join(os.path.abspath(f'{images_path}/sad'), image_name) for image_name in
                      os.listdir(f'{images_path}/sad')]
        surprised_path = [os.path.join(os.path.abspath(f'{images_path}/surprised'), image_name) for image_name in
                      os.listdir(f'{images_path}/surprised')]

        angry_images, happy_images, sad_images, surprised_images = Preprocess.read_images(angry_path),Preprocess.read_images(happy_path),Preprocess.read_images(sad_path),Preprocess.read_images(surprised_path)
        angry_labels,happy_labels, sad_labels, surprised_labels = np.asarray([1]*angry_images.shape[0]),np.asarray([1]*happy_images.shape[0]),np.asarray([1]*sad_images.shape[0]),np.asarray([1]*surprised_images.shape[0])

        return np.concatenate((angry_images, happy_images, sad_images, surprised_images), axis=0),np.concatenate((angry_labels,happy_labels, sad_labels, surprised_labels), axis=0)

    @staticmethod
    def read_vechicle():
        images_path = r'C:\Users\roinis\Documents\Degree\learning\vechicle\vechicle_with_train\vechicles\train'
        labels_names = ['bike','boat','bus','car','cycle','plane','scooty']
        images, labels = [], []
        for index,label in enumerate(labels_names):
            path = f'{images_path}/{label}'
            paths = [os.path.join(path, image_name) for image_name in os.listdir(path)]
            labels.extend([index]*len(paths))
            images.extend(list(Preprocess.read_images(paths)))

        return np.asarray(images),np.asarray(labels)

    @staticmethod
    def read_cellphone():
        images_path = r'C:\Users\roinis\Documents\Degree\learning\cellphone\training\training'
        labels_names = ['cellphone-NO','cellphone-YES']
        images, labels = [], []
        for index,label in enumerate(labels_names):
            path = f'{images_path}/{label}'
            paths = [os.path.join(path, image_name) for image_name in os.listdir(path)]
            labels.extend([index]*len(paths))
            images.extend(list(Preprocess.read_images(paths)))
        return np.asarray(images), np.asarray(labels)


    @staticmethod
    def read_indian_currency():
        images_path = r'C:\Users\roinis\Documents\Degree\learning\indian_currency'
        labels_names = ['1Hundrednote','2Hundrednote','2Thousandnote','5Hundrednote','Fiftynote','Tennote','Twentynote']
        images, labels = [], []
        for index,label in enumerate(labels_names):
            train_path = f'{images_path}/Train/{label}'
            test_path = f'{images_path}/Test/{label}'
            train_paths = [os.path.join(train_path, image_name) for image_name in os.listdir(train_path)]
            test_paths = [os.path.join(test_path, image_name) for image_name in os.listdir(test_path)]

            labels.extend([index]*(len(train_paths) + len(test_paths)))

            images.extend(list(Preprocess.read_images(train_paths)))
            images.extend(list(Preprocess.read_images(test_paths)))
        return np.asarray(images), np.asarray(labels)


class Dataset:
    def __init__(self, X, y,name):
        self.X = X
        self.y = y
        self.train_X = X
        self.test_X = np.empty_like(X)
        self.train_y = y
        self.test_y = np.empty_like(y)
        self.n_classes = np.unique(y).size
        self.name=name

if __name__ == '__main__':
    x = Preprocess.read_indian_currency()
    print(x)