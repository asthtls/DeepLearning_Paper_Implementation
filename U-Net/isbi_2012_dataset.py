# dataset preprocessing

import tifffile as tiff 
import skimage.io as io 
import os 

train_img_folder_path = os.path.join('isbi_2012','preprocessed','train_imgs')
train_label_folder_path = os.path.join('isbi_2012','preprocessed','train_labels')
test_img_folder_path = os.path.join('isbi_2012','preprocessed','test_imgs')

train_imgs = tiff.imread(os.path.join('isbi_2012','raw_data','train-volume.tif'))
train_labels = tiff.imread(os.path.join('isbi_2012','raw_data','train-labels.tif'))
test_imgs = tiff.imread(os.path.join('isbi_2012','raw_data','test-volume.tif'))

print('file shape train_img, train_label, test_img : ', train_imgs.shape, train_labels.shape, test_imgs.shape)

paths = [train_img_folder_path, train_label_folder_path, test_img_folder_path]

for path in paths:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=False)
        print(f"Created directory: {path}")

for img_idx, zip_element in enumerate(zip(train_imgs, train_labels, test_imgs)):
    each_train_img, each_train_label, each_test_img = zip_element

    io.imsave(os.path.join(train_img_folder_path, f"{img_idx}.png"), each_train_img)
    io.imsave(os.path.join(train_label_folder_path, f"{img_idx}.png"), each_train_label)
    io.imsave(os.path.join(test_img_folder_path, f"{img_idx}.png"), each_test_img)
    
        