import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch


def matplotlib_imshow(img):
    assert len(img.shape) == 3
    npimg = img.numpy()
    return (np.transpose(npimg, (1, 2, 0))[:, :, ::-1] * 255).astype(np.uint8)

def trg_to_rgb(self, target):
    assert len(target.shape) == 3
    #
    target = target.to('cpu').softmax(dim=0).argmax(dim=0).to('cpu')
    #
    target = target.detach().cpu().numpy()
    #
    target_rgb = np.zeros_like(target, dtype=np.uint8)
    target_rgb = np.repeat(np.expand_dims(target_rgb[:, :], axis=-1), 3, -1)
    #
    color_table = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (0, 0, 128), 4: (128, 128, 0),
                   5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (0, 64, 64),
                   9: (64, 64, 64), 10: (0, 0, 192), 11: (192, 0, 192), 12: (0, 192, 192),
                   13: (192, 192, 192), 14: (64, 128, 0), 15: (192, 0, 128), 16: (64, 128, 128),
                   17: (192, 128, 128), 18: (128, 64, 0), 19: (128, 192, 0), 20: (0, 64, 128)}
    #
    for i in range(self.cfg['dataset']['num_class']):
        target_rgb[target == i] = np.array(color_table[i])

    return target_rgb
def transform(img, label):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float() / 255.0    # when image is into network -> image pixel value is between 0 and 1
    label = torch.from_numpy(label).float()

    return img, label

if __name__=='__main__':
    train_path = "/storage/sjpark/vehicle_data/Dataset/train_image/"
    ann_path = "/storage/sjpark/vehicle_data/Dataset/ann_train/"
    train_dir = sorted(os.listdir(train_path))
    ann_dir = sorted(os.listdir(ann_path))
    a = os.path.join(train_path, train_dir[0])
    a = Image.open(a)
    a = np.array(a, dtype=np.uint8)

    b = os.path.join(ann_path, ann_dir[0])
    b = Image.open(b, )
    b = np.array(b, dtype=np.uint8)

    a, b = transform(a, b)
    a = matplotlib_imshow(a[0])



