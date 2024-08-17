import numpy as np
import os
import json
import torch
from Config.config import get_config_dict

except_classes = ['motorcycle', 'bicycle', 'twowheeler', 'pedestrian', 'rider', 'sidewalk', 'crosswalk', 'speedbump', 'redlane', 'stoplane', 'trafficlight', 'background']

CLASSES = [
    'background', 'vehicle', 'bus', 'truck', 'policeCar', 'ambulance', 'schoolBus', 'otherCar',
    'freespace', 'curb', 'safetyZone', 'roadMark', 'whiteLane',
    'yellowLane', 'blueLane', 'constructionGuide', 'trafficDrum',
    'rubberCone', 'trafficSign', 'warningTriangle', 'fence']

cfg = get_config_dict()




#------------------------------------------- make_image-rgb--------------------------------------------------------------#
def pred_to_rgb(pred):
    assert len(pred.shape) == 3
    #
    pred = pred.to('cpu').softmax(dim=0).argmax(dim=0).to('cpu')
    #
    pred = pred.detach().cpu().numpy()
    #
    pred_rgb = np.zeros_like(pred, dtype=np.uint8)
    pred_rgb = np.repeat(np.expand_dims(pred_rgb[:, :], axis=-1), 3, -1)
    #
    color_table = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (0, 0, 128), 4: (128, 128, 0),
                   5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (0, 64, 64),
                   9: (64, 64, 64), 10: (0, 0, 192), 11: (192, 0, 192), 12: (0, 192, 192),
                   13: (192, 192, 192), 14: (64, 128, 0), 15: (192, 0, 128), 16: (64, 128, 128),
                   17: (192, 128, 128), 18: (128, 64, 0), 19: (128, 192, 0), 20: (0, 64, 128)}
    #
    for i in range(len(CLASSES)):
        pred_rgb[pred == i] = np.array(color_table[i])

    return pred_rgb


def trg_to_rgb(target):
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
    for i in range(len(CLASSES)):
        target_rgb[target == i] = np.array(color_table[i])

    return target_rgb


def trg_to_class_rgb(target, cls):
    assert len(target.shape) == 3

    for i in range(len(CLASSES)):
        CLASSES[i] = CLASSES[i].lower()
    #
    cls = cls.lower()
    #
    color_table = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (0, 0, 128), 4: (128, 128, 0),
                   5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (0, 64, 64),
                   9: (64, 64, 64), 10: (0, 0, 192), 11: (192, 0, 192), 12: (0, 192, 192),
                   13: (192, 192, 192), 14: (64, 128, 0), 15: (192, 0, 128), 16: (64, 128, 128),
                   17: (192, 128, 128), 18: (128, 64, 0), 19: (128, 192, 0), 20: (0, 64, 128)}

    #
    target = target.to('cpu').softmax(dim=0).argmax(dim=0).to('cpu')
    #
    target = target.detach().cpu().numpy()
    #
    target_rgb = np.zeros_like(target, dtype=np.uint8)
    target_rgb = np.repeat(np.expand_dims(target_rgb[:, :], axis=-1), 3, -1)

    i = CLASSES.index(cls.lower())
    target_rgb[target == i] = np.array(color_table[i])

    return target_rgb


def pred_to_class_rgb(pred, cls):
    assert len(pred.shape) == 3
    #
    for i in range(len(CLASSES)):
        CLASSES[i] = CLASSES[i].lower()
    #
    cls = cls.lower()
    #
    color_table = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (0, 0, 128), 4: (128, 128, 0),
                   5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128), 8: (0, 64, 64),
                   9: (64, 64, 64), 10: (0, 0, 192), 11: (192, 0, 192), 12: (0, 192, 192),
                   13: (192, 192, 192), 14: (64, 128, 0), 15: (192, 0, 128), 16: (64, 128, 128),
                   17: (192, 128, 128), 18: (128, 64, 0), 19: (128, 192, 0), 20: (0, 64, 128)}

    #
    #
    pred = pred.to('cpu').softmax(dim=0).argmax(dim=0).to('cpu')
    #
    pred = pred.detach().cpu().numpy()
    #
    pred_rgb = np.zeros_like(pred, dtype=np.uint8)
    pred_rgb = np.repeat(np.expand_dims(pred_rgb[:, :], axis=-1), 3, -1)
    #
    i = CLASSES.index(cls)
    pred_rgb[pred == i] = np.array(color_table[i])

    return pred_rgb

def matplotlib_imshow(img):
    assert len(img.shape) == 3
    npimg = img.numpy()
    return (np.transpose(npimg, (1, 2, 0))[:, :, ::-1] * 255).astype(np.uint8)

#------------------------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------load_jsonfile------------------------------------------------------#
def load_json_file(idx):
    json_path = '/storage/sjpark/vehicle_data/Dataset/val_json/'
    idx_json = sorted(os.listdir(json_path))
    for i in range(len(except_classes)):
        except_classes[i] = except_classes[i].lower()

    json_file = os.path.join(json_path, idx_json[idx])
    with open(json_file, 'r') as f:
        json_data1 = json.load(f)
    json_data = json_data1['annotations']
    # json_cls_data = json_data1['class']
    ann_json = []
    #
    for i in range(len(json_data)):
        if json_data[i]['class'] in except_classes:
            pass
        else:
            ann_json.append(json_data[i])


    return idx_json[idx], ann_json
#------------------------------------------------------------------------------------------------------------------------#

#------------------------------------------------------calculate_IoU-----------------------------------------------------#
def IoU(pred, target, cls, thr=0.5, dim=(2, 3), epsilon=0.001):
    #
    cls = cls.lower()
    #
    ious = {}
    y_true = target.to(torch.float32)
    y_pred = pred.to(torch.float32)

    inter = (y_true * y_pred).sum(dim=(0, 1))
    union = (y_true + y_pred - y_true * y_pred).sum(dim=(0, 1))

    iou = ((inter + epsilon) / (union + epsilon)).mean()
    ious[cls] = iou
    return ious

def make_bbox(file, json_path, target_image, pred_image):
    ious = []
    org_cls = []
    #
    for i in range(len(CLASSES)):
        CLASSES[i] = CLASSES[i].lower()
    #
    org_res = (1920, 1080)
    target_res = cfg['dataset']['size']
    #
    scale_x = target_res[0] / org_res[0]
    scale_y = target_res[1] / org_res[1]
    count = 0
    for i in range(len(json_path)):
        polygon = json_path[i]['polygon']
        cls = json_path[i]['class'].lower()
        if cls in except_classes:
            pass
        else:
            for j in range(len(polygon)):
                if j % 2 == 0:
                    polygon[j] = polygon[j] * scale_x
                else:
                    polygon[j] = polygon[j] * scale_y

            polygon = np.array(polygon, np.int32).reshape(-1, 2)
            if polygon.size == 0:
                pass
            else:
                x_min = np.min(polygon[:, 0])
                y_min = np.min(polygon[:, 1])
                x_max = np.max(polygon[:, 0])
                y_max = np.max(polygon[:, 1])
                if (x_min == x_max) or (y_min == y_max):
                    pass
                else:
                    # make Class index
                    if cls not in CLASSES:
                        print("error")
                    else:
                        x = CLASSES.index(cls)
                    #
                    crop_target_image = target_image[:, y_min:y_max:, x_min:x_max]
                    crop_target_image[crop_target_image != x] = 0
                    #
                    crop_pred_image = pred_image[:, y_min:y_max:, x_min:x_max]
                    crop_pred_image[crop_pred_image != x] = 0
                    #
                    crop_target_image = torch.where(crop_target_image >= 1, torch.tensor(1.0), torch.tensor(0.0))
                    crop_pred_image = torch.where(crop_pred_image >= 1, torch.tensor(1.0), torch.tensor(0.0))
                    org_cls.append(cls)
                    iou = IoU(crop_pred_image, crop_target_image, cls)
                    ious.append(iou)

    return ious

#------------------------------------------------------------------------------------------------------------------------#


#------------------------------------------------------Image_crop--------------------------------------------------------#
def crop_image(target, pred, json_path):
    target_image_list = []
    pred_image_list = []
    cls_list = []
    count = 0
    #
    for i in range(len(json_path)):
        polygon = json_path[i]['polygon']
        cls = json_path[i]['class']
        polygon = np.array(polygon, np.int32).reshape(-1, 2)
        if polygon.size == 0:
            pass
        else:
            x_min = np.min(polygon[:, 0]) - 20
            if x_min < 0:
                x_min = 0
            #
            y_min = np.min(polygon[:, 1]) - 20
            if y_min < 0:
                y_min = 0
            #
            x_max = np.max(polygon[:, 0]) + 20
            if x_max > cfg['dataset']['image_size']:
                x_max = cfg['dataset']['image_size']
            #
            y_max = np.max(polygon[:, 1]) + 20
            if y_max > cfg['dataset']['image_size']:
                y_max = cfg['dataset']['image_size']
            #
            if (x_min == x_max) or (y_min == y_max):
                pass
            else:
                crop_target_image = target[:, y_min:y_max:, x_min:x_max]
                crop_pred_image = pred[:, y_min:y_max:, x_min:x_max]
                target_image_list.append(crop_target_image)
                pred_image_list.append(crop_pred_image)
                cls_list.append(cls)

    if len(cls_list) != len(target_image_list):
        print("error")

    return target_image_list, pred_image_list, cls_list
#------------------------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------Calculate_pixel_Acc------------------------------------------------#
def pixel_acc_cls(pred, target, cls):
    #
    for i in range(len(CLASSES)):
        CLASSES[i] = CLASSES[i].lower()
    #
    for j in range(len(except_classes)):
        except_classes[j] = except_classes[j].lower()
    #
    for j in range(len(cls)):
        cls[j] = cls[j].lower()
    #
    class_acc = []
    for c in cls:
        if c in except_classes:
            pass
        else:
            index = CLASSES.index(c)
            cls_mask = (target == index)
            correct = (pred[cls_mask] == index).sum()
            total = cls_mask.sum()

            class_acc.append(correct / total)

    return class_acc
#------------------------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------Calculate_precison_recall-------------------------------------------#
def precision_recall(target, pred):
    confusion_matrix = torch.zeros((20, 20))

    for true_class in range(20):
        for pred_class in range(20):
            confusion_matrix[true_class, pred_class] = torch.sum((target == true_class) & (pred == pred_class))

    precision = np.zeros(20)
    recall = np.zeros(20)

    for cls in range(20):
        tp = confusion_matrix[cls, cls]
        fp = torch.sum(confusion_matrix[:, cls]) - tp
        fn = torch.sum(confusion_matrix[cls, :]) - tp
        precision[cls] = tp / (tp + fp + 1e-5)
        recall[cls] = tp / (tp + fn + 1e-5)



    return precision, recall
#------------------------------------------------------------------------------------------------------------------------#