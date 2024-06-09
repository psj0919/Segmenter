import os
import random
import shutil

def cross_validation(train, label):
    dst_train_path = '/storage/sjpark/vehicle_data/Dataset/val_image'
    dst_label_path = '/storage/sjpark/vehicle_data/Dataset/ann_val'


    img_list = (random.sample(train, k = 364))


    for i in img_list:
        train_path = os.path.join('/storage/sjpark/vehicle_data/Dataset/train_image', i)
        i = i.split('.')[0] + '.png'
        label_path = os.path.join('/storage/sjpark/vehicle_data/Dataset/ann_train', i)

        shutil.move(train_path, dst_train_path)
        shutil.move(label_path, dst_label_path)







if __name__=='__main__':
    train_path = os.listdir('/storage/sjpark/vehicle_data/Dataset/train_image')
    label_path = os.listdir('/storage/sjpark/vehicle_data/Dataset/ann_train')
    # print("51392 --> number of trian_image")
    # print("12848 --> number of val_image")
    # cross_validation(train_path, label_path)

    print(len(train_path))
    print(len(label_path))
    print(len(os.listdir('/storage/sjpark/vehicle_data/Dataset/val_image')))
    print(len(os.listdir('/storage/sjpark/vehicle_data/Dataset/ann_val')))