import os
import shutil


def move_dataset(sub_path):
    dir_list = os.listdir(sub_path)
    for i in dir_list:
        path = os.path.join(sub_path, i+'/sensor_raw_data/camera')
        for subdir, _, files in os.walk(path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    src_path = os.path.join(subdir, file)
                    dst_path = os.path.join('/storage/sjpark/vehicle_data/Dataset/val_image', file)
                    shutil.copy(src_path, dst_path)


def total_image(sub_path):
    pre_images = 0
    cur_imges = 0

    # pre_num_image
    dir_list = os.listdir(sub_path)
    for i in dir_list:
        path = os.path.join(sub_path, i + '/sensor_raw_data/camera')
        for subdir, _, files in os.walk(path):
            for file in files:
                pre_images += 1

    cur_imges = len(os.listdir('/storage/sjpark/vehicle_data/Dataset/val_image'))

    print("pre_num_images: {}".format(pre_images))
    print("cur_num_images: {}".format(cur_imges))







if __name__=='__main__':
    # train_image
    sub_path = '/storage/sjpark/vehicle_data/Dataset/original_img/img_train'
    move_dataset(sub_path)
    total_image(sub_path)

    # validation_image
    sub_path = '/storage/sjpark/vehicle_data/Dataset/original_img/img_val/'
    move_dataset(sub_path)
    total_image(sub_path)





