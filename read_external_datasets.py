import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import os


def read_flowers_colors():
    meta_df_path = r"C:\Users\roinis\Documents\Degree\learning\flowers\flower_images\flower_images\flower_labels.csv"
    meta_df = pd.read_csv(meta_df_path)

    updated_path = r'C:\Users\roinis\Documents\Degree\learning\flowers\flower_images\flower_images\\'

    meta_df["file_path"] = updated_path + meta_df["file"]
    meta_df.drop(columns=['file'], inplace=True)

    images_arr = []
    labels = []
    for index, img in meta_df.iterrows():
        image_path = img["file_path"]
        image = Image.open(image_path).convert("RGB")
        image.load()
        image = image.resize((32, 32), 3)
        pixels = np.asarray(image, dtype="int32")
        images_arr.append(pixels)
        labels.append(img["label"])

    return np.asarray(images_arr), np.asarray(labels)


def read_gemstones_dataset():
    train_gemstone_path = r'C:\Users\roinis\Documents\Degree\learning\gemstones\train'
    test_gemstone_path = r'C:\Users\roinis\Documents\Degree\learning\gemstones\test'

    images = []
    labels_category = []
    labels = []

    for label in os.listdir(train_gemstone_path):
        for img in os.listdir(train_gemstone_path + '\\' + label):
            image = Image.open(train_gemstone_path + '\\' + label + '\\' + img).convert("RGB")
            image.load()
            image = image.resize((32, 32), Image.ANTIALIAS)
            pixels = np.asarray(image, dtype="int32")
            images.append(pixels)
            labels_category.append(label)

    for label in os.listdir(test_gemstone_path):
        for img in os.listdir(test_gemstone_path + '\\' + label):
            image = Image.open(test_gemstone_path + '\\' + label + '\\' + img).convert("RGB")
            image.load()
            image = image.resize((32, 32), Image.ANTIALIAS)
            pixels = np.asarray(image, dtype="int32")
            images.append(pixels)
            labels_category.append(label)

    labels_string = {label: index for index, label in enumerate(np.unique(labels_category))}
    for label in labels_category:
        labels.append(labels_string[label])
    return np.asarray(images), np.asarray(labels)


def read_brain_mri_dataset():
    brain_mri_path = r'C:\Users\roinis\Documents\Degree\learning\brain_mri'

    images = []
    labels_category = []
    labels = []

    for label in os.listdir(brain_mri_path):
        for img in os.listdir(brain_mri_path + '\\' + label):
            image = Image.open(brain_mri_path + '\\' + label + '\\' + img).convert("RGB")
            image.load()
            image = image.resize((32, 32), Image.ANTIALIAS)
            pixels = np.asarray(image, dtype="int32")
            images.append(pixels)
            labels_category.append(label)

    labels_string = {label: index for index, label in enumerate(np.unique(labels_category))}
    for label in labels_category:
        labels.append(labels_string[label])
    return np.asarray(images), np.asarray(labels)


def read_covid_xray_dataset():
    covid_xray_path = r'C:\Users\roinis\Documents\Degree\learning\covid_chest\dataset'

    images = []
    labels_category = []
    labels = []

    for label in os.listdir(covid_xray_path):
        for img in os.listdir(covid_xray_path + '\\' + label):
            image = Image.open(covid_xray_path + '\\' + label + '\\' + img).convert("RGB")
            image.load()
            image = image.resize((32, 32), Image.ANTIALIAS)
            pixels = np.asarray(image, dtype="int32")
            images.append(pixels)
            labels_category.append(label)

    labels_string = {label: index for index, label in enumerate(np.unique(labels_category))}
    for label in labels_category:
        labels.append(labels_string[label])
    return np.asarray(images), np.asarray(labels)


def read_hurricane_damage_dataset():
    hurricane_damage_path = r'C:\Users\roinis\Documents\Degree\learning\hurricane_damage'

    images = []
    labels_category = []
    labels = []

    for dir in os.listdir(hurricane_damage_path):
        for label in os.listdir(hurricane_damage_path + '\\' + dir):
            for img in os.listdir(hurricane_damage_path + '\\' + dir + '\\' + label):
                image = Image.open(hurricane_damage_path + '\\' + dir + '\\' + label + '\\' + img).convert("RGB")
                image.load()
                image = image.resize((32, 32), Image.ANTIALIAS)
                pixels = np.asarray(image, dtype="int32")
                images.append(pixels)
                labels_category.append(label)

    labels_string = {label: index for index, label in enumerate(np.unique(labels_category))}
    for label in labels_category:
        labels.append(labels_string[label])
    return np.asarray(images), np.asarray(labels)


def read_vegetables_dataset():
    vegetables_path = r'C:\Users\roinis\Documents\Degree\learning\vagetables\training_images'

    images = []
    labels_category = []
    labels = []

    for img in os.listdir(vegetables_path):
        if not '.xml' in img:
            label_list = img.split('_')
            label = label_list[0]
            image = Image.open(vegetables_path + '\\' + img).convert("RGB")
            image.load()
            image = image.resize((32, 32), Image.ANTIALIAS)
            pixels = np.asarray(image, dtype="int32")
            images.append(pixels)
            labels_category.append(label)

    labels_string = {label: index for index, label in enumerate(np.unique(labels_category))}
    for label in labels_category:
        labels.append(labels_string[label])
    return np.asarray(images), np.asarray(labels)


def read_clothes_colors_dataset():
    clothes_colors_path = r'C:\Users\roinis\Documents\Degree\learning\cloth_colors\train'

    images = []
    labels_category = []
    labels = []

    for label in os.listdir(clothes_colors_path):
        for img in os.listdir(clothes_colors_path + '\\' + label):
            image = Image.open(clothes_colors_path + '\\' + label + '\\' + img).convert("RGB")
            image.load()
            image = image.resize((32, 32), Image.ANTIALIAS)
            pixels = np.asarray(image, dtype="int32")
            images.append(pixels)
            labels_category.append(label)

    labels_string = {label: index for index, label in enumerate(np.unique(labels_category))}
    for label in labels_category:
        labels.append(labels_string[label])
    return np.asarray(images), np.asarray(labels)


def read_clean_messy_room_dataset():
    rooms_path = r'C:\Users\roinis\Documents\Degree\learning\clean_messy_room\images\train'

    images = []
    labels_category = []
    labels = []

    for label in os.listdir(rooms_path):
        for img in os.listdir(rooms_path + '\\' + label):
            image = Image.open(rooms_path + '\\' + label + '\\' + img).convert("RGB")
            image.load()
            image = image.resize((32, 32), Image.ANTIALIAS)
            pixels = np.asarray(image, dtype="int32")
            images.append(pixels)
            labels_category.append(label)

    labels_string = {label: index for index, label in enumerate(np.unique(labels_category))}
    for label in labels_category:
        labels.append(labels_string[label])
    return np.asarray(images), np.asarray(labels)


def read_traffic_recognition_dataset():
    traffic_recognition_path = r'C:\Users\roinis\Documents\Degree\learning\traffic_sign\crop_dataset\crop_dataset'

    images = []
    labels = []

    for label in os.listdir(traffic_recognition_path):
        for img in os.listdir(traffic_recognition_path + '\\' + label):
            image = Image.open(traffic_recognition_path + '\\' + label + '\\' + img).convert("RGB")
            image.load()
            image = image.resize((32, 32), Image.ANTIALIAS)
            pixels = np.asarray(image, dtype="int32")
            images.append(pixels)
            labels.append(label[3:])

    return np.asarray(images), np.asarray(labels)


def read_sign_language_dataset():
    sign_language_path = r'C:\Users\roinis\Documents\Degree\learning\sign_lang\asl_dataset'

    images = []
    labels_category = []
    labels = []

    for label in os.listdir(sign_language_path):
        for img in os.listdir(sign_language_path + '\\' + label):
            image = Image.open(sign_language_path + '\\' + label + '\\' + img).convert("RGB")
            image.load()
            image = image.resize((32, 32), Image.ANTIALIAS)
            pixels = np.asarray(image, dtype="int32")
            images.append(pixels)
            labels_category.append(label)

    labels_string = {label: index for index, label in enumerate(np.unique(labels_category))}
    for label in labels_category:
        labels.append(labels_string[label])
    return np.asarray(images), np.asarray(labels)


def read_shapes_dataset():
    shapes_path = r'C:\Users\roinis\Documents\Degree\learning\sahpes\shapes'

    images = []
    labels_category = []
    labels = []

    for label in os.listdir(shapes_path):
        for img in os.listdir(shapes_path + '\\' + label):
            image = Image.open(shapes_path + '\\' + label + '\\' + img).convert("RGB")
            image.load()
            image = image.resize((32, 32), Image.ANTIALIAS)
            pixels = np.asarray(image, dtype="int32")
            images.append(pixels)
            labels_category.append(label)

    labels_string = {label: index for index, label in enumerate(np.unique(labels_category))}
    for label in labels_category:
        labels.append(labels_string[label])
    return np.asarray(images), np.asarray(labels)


# brain_tumor
def read_images(images_path):
    images_arr = []
    for image_path in images_path:
        image = Image.open(image_path).convert('RGB')
        image.load()
        image = image.resize((32, 32), Image.ANTIALIAS)
        pixels = np.asarray(image, dtype="int32")

        images_arr.append(pixels)

    return np.asarray(images_arr)


def read_brain_tumor_dataset():
    images_path = r'C:\Users\roinis\Documents\Degree\learning\brain_tumor'
    labels_df = pd.read_csv(f'{images_path}/Brain Tumor.csv')
    labels_df['image_path'] = f'{images_path}/Brain Tumor/Brain Tumor/' + labels_df['Image'] + '.jpg'
    # folders =['test','test_another','train_another','validation_another']
    images = read_images(list(labels_df['image_path']))
    labels = np.asarray(list(labels_df['Class']))
    return images, labels


def read_wildfire_dataset():
    wildfire_path = r'C:\Users\roinis\Documents\Degree\learning\wildfire\forest_fire\Training and Validation'

    images = []
    labels_category = []
    labels = []

    for label in os.listdir(wildfire_path):
        for img in os.listdir(wildfire_path + '\\' + label):
            image = Image.open(wildfire_path + '\\' + label + '\\' + img).convert("RGB")
            image.load()
            image = image.resize((32, 32), Image.ANTIALIAS)
            pixels = np.asarray(image, dtype="int32")
            images.append(pixels)
            labels_category.append(label)

    labels_string = {label: index for index, label in enumerate(np.unique(labels_category))}
    for label in labels_category:
        labels.append(labels_string[label])
    return np.asarray(images), np.asarray(labels)


def read_tom_jerry_dataset():
    images_path = r'C:\Users\roinis\Documents\Degree\learning\tom_jerry\train\train'
    angry_path = [os.path.join(os.path.abspath(f'{images_path}/angry'), image_name) for image_name in
                  os.listdir(f'{images_path}/angry')]
    happy_path = [os.path.join(os.path.abspath(f'{images_path}/happy'), image_name) for image_name in
                  os.listdir(f'{images_path}/happy')]
    sad_path = [os.path.join(os.path.abspath(f'{images_path}/sad'), image_name) for image_name in
                os.listdir(f'{images_path}/sad')]
    surprised_path = [os.path.join(os.path.abspath(f'{images_path}/surprised'), image_name) for image_name in
                      os.listdir(f'{images_path}/surprised')]

    angry_images, happy_images, sad_images, surprised_images = read_images(angry_path), read_images(
        happy_path), read_images(sad_path), read_images(surprised_path)
    angry_labels, happy_labels, sad_labels, surprised_labels = np.asarray([0] * angry_images.shape[0]), np.asarray(
        [1] * happy_images.shape[0]), np.asarray([2] * sad_images.shape[0]), np.asarray([3] * surprised_images.shape[0])

    return np.concatenate((angry_images, happy_images, sad_images, surprised_images), axis=0), np.concatenate(
        (angry_labels, happy_labels, sad_labels, surprised_labels), axis=0)


def read_english_characters_dataset():
    meta_df_path = r"C:\Users\roinis\Documents\Degree\learning\english_char\english.csv"
    meta_df = pd.read_csv(meta_df_path)

    updated_path = r'C:\Users\roinis\Documents\Degree\learning\english_char\\'

    meta_df["file_path"] = updated_path + meta_df["image"]
    meta_df.drop(columns=['image'], inplace=True)

    images_arr = []
    labels = []

    for index, img in meta_df.iterrows():
        image_path = img["file_path"]
        image = Image.open(image_path).convert("RGB")
        image.load()
        image = image.resize((32, 32), 3)
        pixels = np.asarray(image, dtype="int32")
        images_arr.append(pixels)
        try:
            label = int(img["label"])
        except Exception:
            label = int(ord(img["label"]))
        labels.append(label)

    return np.asarray(images_arr), np.asarray(labels)


def read_bows_dataset():
    bows_path = r'C:\Users\roinis\Documents\Degree\learning\bows\bow'

    images = []
    labels_category = []
    labels = []

    for label in os.listdir(bows_path):
        for img in os.listdir(bows_path + '\\' + label):
            image = Image.open(bows_path + '\\' + label + '\\' + img).convert("RGB")
            image.load()
            image = image.resize((32, 32), Image.ANTIALIAS)
            pixels = np.asarray(image, dtype="int32")
            images.append(pixels)
            labels_category.append(label)

    labels_string = {label: index for index, label in enumerate(np.unique(labels_category))}
    for label in labels_category:
        labels.append(labels_string[label])
    return np.asarray(images), np.asarray(labels)


def read_hand_gestures_dataset():
    bows_path = r'C:\Users\roinis\Documents\Degree\learning\hand_gestures'

    images = []
    labels_category = []
    labels = []

    for label in os.listdir(bows_path):
        for img in os.listdir(bows_path + '\\' + label):
            image = Image.open(bows_path + '\\' + label + '\\' + img).convert("RGB")
            image.load()
            image = image.resize((32, 32), Image.ANTIALIAS)
            pixels = np.asarray(image, dtype="int32")
            images.append(pixels)
            labels_category.append(label)

    labels_string = {label: index for index, label in enumerate(np.unique(labels_category))}
    for label in labels_category:
        labels.append(labels_string[label])
    return np.asarray(images), np.asarray(labels)

def read_iris_dataset():
    iris_path = r'C:\Users\roinis\Documents\Degree\learning\iris'

    images = []
    labels_category = []
    labels = []

    for label in os.listdir(iris_path):
        for img in os.listdir(iris_path + '\\' + label):
            image = Image.open(iris_path + '\\' + label + '\\' + img).convert("RGB")
            image.load()
            image = image.resize((32, 32), Image.ANTIALIAS)
            pixels = np.asarray(image, dtype="int32")
            images.append(pixels)
            labels_category.append(label)

    labels_string = {label: index for index, label in enumerate(np.unique(labels_category))}
    for label in labels_category:
        labels.append(labels_string[label])
    return np.asarray(images), np.asarray(labels)


def read_vehicles_dataset():
    images_path = r'C:\Users\roinis\Documents\Degree\learning\vechicle\vechicle_with_train\vechicles\train'
    labels_names = ['bike', 'boat', 'bus', 'car', 'cycle', 'plane', 'scooty']
    images, labels = [], []
    for index, label in enumerate(labels_names):
        path = f'{images_path}/{label}'
        paths = [os.path.join(path, image_name) for image_name in os.listdir(path)]
        labels.extend([index] * len(paths))
        images.extend(list(read_images(paths)))

    return np.asarray(images), np.asarray(labels)


def read_cellphone_dataset():
    images_path = r'C:\Users\roinis\Documents\Degree\learning\cellphone\training\training'
    labels_names = ['cellphone-NO', 'cellphone-YES']
    images, labels = [], []
    for index, label in enumerate(labels_names):
        path = f'{images_path}/{label}'
        paths = [os.path.join(path, image_name) for image_name in os.listdir(path)]
        labels.extend([index] * len(paths))
        images.extend(list(read_images(paths)))

    return np.asarray(images), np.asarray(labels)


def read_indian_currency():
    images_path = r'C:\Users\roinis\Documents\Degree\learning\indian_currency'
    labels_names = ['1Hundrednote', '2Hundrednote', '2Thousandnote', '5Hundrednote', 'Fiftynote', 'Tennote',
                    'Twentynote']
    images, labels = [], []
    for index, label in enumerate(labels_names):
        train_path = f'{images_path}/Train/{label}'
        test_path = f'{images_path}/Test/{label}'
        train_paths = [os.path.join(train_path, image_name) for image_name in os.listdir(train_path)]
        test_paths = [os.path.join(test_path, image_name) for image_name in os.listdir(test_path)]

        labels.extend([index] * (len(train_paths) + len(test_paths)))

        images.extend(list(read_images(train_paths)))
        images.extend(list(read_images(test_paths)))
    return np.asarray(images), np.asarray(labels)


if __name__ == '__main__':
    read_indian_currency()
    # read_gemstones_dataset()
    # read_hurricane_damage_dataset()
