import numpy as np
import os
import json
import torch
import matplotlib.pyplot as plt


from Config.config import get_config_dict

except_classes = ['motorcycle', 'bicycle', 'twowheeler', 'pedestrian', 'rider', 'sidewalk', 'crosswalk', 'speedbump', 'redlane', 'stoplane', 'trafficlight']

CLASSES = [
    'background', 'vehicle', 'bus', 'truck', 'policeCar', 'ambulance', 'schoolBus', 'otherCar',
    'freespace', 'curb', 'safetyZone', 'roadMark', 'whiteLane',
    'yellowLane', 'blueLane', 'constructionGuide', 'trafficDrum',
    'rubberCone', 'trafficSign', 'warningTriangle', 'fence']


cfg = get_config_dict()
if cfg['model']['mode'] == 'train':
    pass
else:
    from Config.config_test import get_test_config_dict
    cfg = get_test_config_dict()


#------------------------------------------- make_image-rgb--------------------------------------------------------------#
def pred_to_rgb(pred):
    assert len(pred.shape) == 3
    #
    pred = pred.softmax(dim=0).argmax(dim=0).to('cpu')
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
    target = target.softmax(dim=0).argmax(dim=0).to('cpu')
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
    target = target.softmax(dim=0).argmax(dim=0).to('cpu')
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
    pred = pred.softmax(dim=0).argmax(dim=0).to('cpu')
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
    if cfg['model']['mode'] == 'train':
        json_path = '/storage/sjpark/vehicle_data/Dataset/val_json/'
    elif cfg['model']['mode'] == 'test':
        json_path = '/storage/sjpark/vehicle_data/Dataset/test_json/'

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
def IoU(pred, target, cls, eps = 1e-5):
    #
    cls = cls.lower()
    #
    ious = {}
    y_true = target.to(torch.float32)
    y_pred = pred.to(torch.float32)

    inter = (y_true * y_pred).sum(dim=(1, 2))
    union = (y_true + y_pred - y_true * y_pred).sum(dim=(1, 2))


    iou = (inter / (union + eps)).mean()

    ious[cls] = iou

    return ious

def make_bbox(json_path, target_image, pred_image):
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
    #
    for i in range(len(json_path)):
        pred = pred_image.clone()
        target = target_image.clone()
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
                    if x == 0:
                        crop_target_image = target[:, y_min:y_max:, x_min:x_max].clone()
                        crop_target_image[crop_target_image != x] = 1
                        #
                        crop_pred_image = pred[:, y_min:y_max:, x_min:x_max].clone()
                        crop_pred_image[crop_pred_image != x] = 1
                        crop_target_image = torch.where(crop_target_image == 0, torch.tensor(1.0), torch.tensor(0.0))
                        crop_pred_image = torch.where(crop_pred_image == 0, torch.tensor(1.0), torch.tensor(0.0))

                    else:
                        crop_target_image = target[:, y_min:y_max:, x_min:x_max].clone()
                        crop_target_image[crop_target_image != x] = 0
                        #
                        crop_pred_image = pred[:, y_min:y_max:, x_min:x_max].clone()
                        crop_pred_image[crop_pred_image != x] = 0
                        #
                        crop_target_image = torch.where(crop_target_image >= 1, torch.tensor(1.0), torch.tensor(0.0))
                        crop_pred_image = torch.where(crop_pred_image >= 1, torch.tensor(1.0), torch.tensor(0.0))
                    #
                    iou = IoU(crop_pred_image, crop_target_image, cls)
                    #
                    for key, val in iou.items():
                        org_cls.append(key)
                    #
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
        cls = json_path[i]['class'].lower()
        if cls in except_classes:
            pass
        else:
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


    return target_image_list, pred_image_list, cls_list
#------------------------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------Calculate_pixel_Acc------------------------------------------------#
def pixel_acc_cls(pred, target, json_path, eps = 1e-5):
    accs = []
    class_acc = {}

    #
    for i in range(len(CLASSES)):
        CLASSES[i] = CLASSES[i].lower()
    #
    for j in range(len(except_classes)):
        except_classes[j] = except_classes[j].lower()
    #
    for i in range(len(json_path)):
        polygon = json_path[i]['polygon']
        cls = json_path[i]['class'].lower()
        if cls in except_classes:
            pass
        else:
            polygon = np.array(polygon, np.int32).reshape(-1, 2)
            if polygon.size == 0:
                pass
            else:
                x_min = np.min(polygon[:, 0])
                y_min = np.min(polygon[:, 1])
                x_max = np.max(polygon[:, 0])
                y_max = np.max(polygon[:, 1])
            #
                crop_target_image = target[y_min:y_max:, x_min:x_max].clone()
                crop_pred_image = pred[y_min:y_max:, x_min:x_max].clone()
                #
                index = CLASSES.index(cls)
                cls_mask = (crop_target_image == index)
                correct = (crop_pred_image[cls_mask] == index).sum()
                total = cls_mask.sum()
                acc = correct / (total + eps)
                class_acc.setdefault(cls, []).append(acc)

    return class_acc
#------------------------------------------------------------------------------------------------------------------------#


#-----------------------------------------------------Calculate_precison_recall-------------------------------------------#
def precision_recall(target, pred,  json_path, threshold, eps = 1e-5):
    y_pred = []
    y_true = []
    #
    precision = {}
    recall = {}
    count = 0

    #
    for i in range(len(CLASSES)):
        CLASSES[i] = CLASSES[i].lower()
    #
    for j in range(len(except_classes)):
        except_classes[j] = except_classes[j].lower()
    #
    for i in range(len(json_path)):

        polygon = json_path[i]['polygon']
        cls = json_path[i]['class'].lower()
        if cls in except_classes:
            pass
        else:
            polygon = np.array(polygon, np.int32).reshape(-1, 2)
            if polygon.size == 0:
                count = count - 1
                pass
            else:
                x_min = np.min(polygon[:, 0])
                y_min = np.min(polygon[:, 1])
                x_max = np.max(polygon[:, 0])
                y_max = np.max(polygon[:, 1])
                #
                for p_thr in threshold:
                    crop_target_image = target[:, y_min:y_max:, x_min:x_max].clone()
                    crop_pred_image = pred[:, y_min:y_max:, x_min:x_max].clone()
                    c = CLASSES.index(cls)

                    if c == 0:
                        y_pred.append(torch.where(crop_pred_image[0] > p_thr, 0, 1).clone())
                        y_true.append(torch.where(crop_target_image[0] == 1, 0, 1).clone())
                    else:
                        y_pred.append(torch.where(crop_pred_image[c] > p_thr, c, 0).clone())
                        y_true.append(torch.where(crop_target_image[c] == 1, c, 0).clone())
                    #

                    tp = torch.sum(torch.logical_and(y_pred[count] == c, y_true[count] == c))
                    #
                    if c == 0:
                        fp = torch.sum(torch.logical_and(y_pred[count] == c, y_true[count] == 1))
                    else:
                        fp = torch.sum(torch.logical_and(y_pred[count] == c, y_true[count] == 0))
                    #
                    if c == 0:
                        fn = torch.sum(torch.logical_and(y_pred[count] == 1, y_true[count] == c))
                    else:
                        fn = torch.sum(torch.logical_and(y_pred[count] == 0, y_true[count] == c))

                    pre = tp / (tp + fp + eps)
                    rec = tp / (tp + fn + eps)


                    precision.setdefault(p_thr, {}).setdefault(cls, []).append(pre)
                    recall.setdefault(p_thr, {}).setdefault(cls, []).append(rec)
                    count = count + 1
                    crop_target_image = torch.tensor([])
                    crop_pred_image= torch.tensor([])



    return precision, recall
#------------------------------------------------------------------------------------------------------------------------#
def get_precision(target, pred, json_path, threshold, eps=1e-5):
    y_pred = []
    y_true = []
    #
    precision = {}
    count = 0

    #
    for i in range(len(CLASSES)):
        CLASSES[i] = CLASSES[i].lower()
    #
    for j in range(len(except_classes)):
        except_classes[j] = except_classes[j].lower()
    #
    for i in range(len(json_path)):
        polygon = json_path[i]['polygon']
        cls = json_path[i]['class'].lower()
        if cls in except_classes:
            pass
        else:
            polygon = np.array(polygon, np.int32).reshape(-1, 2)
            if polygon.size == 0:
                count = count - 1
                pass
            else:
                x_min = np.min(polygon[:, 0])
                y_min = np.min(polygon[:, 1])
                x_max = np.max(polygon[:, 0])
                y_max = np.max(polygon[:, 1])
                #
                crop_target_image = target[:, y_min:y_max:, x_min:x_max].clone()
                crop_pred_image = pred[:, y_min:y_max:, x_min:x_max].clone()
                #
                c = CLASSES.index(cls)

                if c == 0:
                    y_pred.append(torch.where(crop_pred_image[0] > threshold, 0, 1).clone())
                    y_true.append(torch.where(crop_target_image[0] == 1, 0, 1).clone())
                else:
                    y_pred.append(torch.where(crop_pred_image[c] > threshold, c, 0).clone())
                    y_true.append(torch.where(crop_target_image[c] == 1, c, 0).clone())
                #

                tp = torch.sum(torch.logical_and(y_pred[count] == c, y_true[count] == c))
                #
                if c == 0:
                    fp = torch.sum(torch.logical_and(y_pred[count] == c, y_true[count] == 1))
                else:
                    fp = torch.sum(torch.logical_and(y_pred[count] == c, y_true[count] == 0))

                pre = tp / (tp + fp + eps)

                precision.setdefault(cls, []).append(pre)
                count = count + 1


    for key, val in precision.items():
        if len(val) > 1:
            precision[key] = sum(val) / len(val)
        else:
            precision[key] = val[0]

    return precision


def get_recall(target, pred, json_path, threshold, eps=1e-5):
    y_pred = []
    y_true = []
    #
    recall = {}
    count = 0

    #
    for i in range(len(CLASSES)):
        CLASSES[i] = CLASSES[i].lower()
    #
    for j in range(len(except_classes)):
        except_classes[j] = except_classes[j].lower()
    #
    for i in range(len(json_path)):
        polygon = json_path[i]['polygon']
        cls = json_path[i]['class'].lower()
        if cls in except_classes:
            pass
        else:
            polygon = np.array(polygon, np.int32).reshape(-1, 2)
            if polygon.size == 0:
                count = count - 1
                pass
            else:
                x_min = np.min(polygon[:, 0])
                y_min = np.min(polygon[:, 1])
                x_max = np.max(polygon[:, 0])
                y_max = np.max(polygon[:, 1])
                #
                crop_target_image = target[:, y_min:y_max:, x_min:x_max].clone()
                crop_pred_image = pred[:, y_min:y_max:, x_min:x_max].clone()
                #
                c = CLASSES.index(cls)

                if c == 0:
                    y_pred.append(torch.where(crop_pred_image[0] > threshold, 0, 1).clone())
                    y_true.append(torch.where(crop_target_image[0] == 1, 0, 1).clone())
                else:
                    y_pred.append(torch.where(crop_pred_image[c] > threshold, c, 0).clone())
                    y_true.append(torch.where(crop_target_image[c] == 1, c, 0).clone())
                #
                tp = torch.sum(torch.logical_and(y_pred[count] == c, y_true[count] == c))
                #
                if c == 0:
                    fn = torch.sum(torch.logical_and(y_pred[count] == 1, y_true[count] == c))
                else:
                    fn = torch.sum(torch.logical_and(y_pred[count] == 0, y_true[count] == c))

                rec = tp / (tp + fn + eps)

                recall.setdefault(cls, []).append(rec)
                count = count + 1

    for key, val in recall.items():
        if len(val) > 1:
            recall[key] = sum(val) / len(val)
        else:
            recall[key] = val[0]

    return recall

#--------------------------------------------precision-recall curve------------------------------------------------------#

def precision_recall_curve(target, pred,  json_path, cls, idx):
    threshold = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    precision = []
    recall = []
    x = {}
    y = {}
    #
    for i in threshold:
        precision.append(get_precision(target, pred, json_path, i))
        recall.append(get_recall(target, pred, json_path, i))
    #
    for i in range(len(cls)):
        for j in range(len(precision)):
            y.setdefault(cls[i], []).append(precision[j][cls[i]].cpu())
            x.setdefault(cls[i], []).append(recall[j][cls[i]].cpu())

    for i in range(len(cls)):
        fig = plt.figure(figsize=(9, 6))
        plt.plot(x[i][cls], y[i][cls])
        plt.scatter(x[i][cls], y[i][cls])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.savefig('/storage/sjpark/vehicle_data/curve/FCN/FCN8s/512/{}/{}'.format(cls[i], cls[i] + '_' + str(int(idx))) +'.png')

        plt.close()


def class_per_histogram(acc, iou, precision, recall):
    accs = []
    ious = []
    precisions = []
    recalls = []
    cls = []
    for key, val in acc.items():
        cls.append(key)

    for i in range(len(cls)):
        accs.append(acc[cls[i]])
        plt.figure()
        plt.hist(accs, label=cls[i] + '_acc')
        plt.legend()
        plt.savefig('/storage/sjpark/vehicle_data/histogram/FCN/FCN8s/256/{}/{}'.format(cls[i], cls[i] + '_Pixel_Accuracy'))
        plt.close()
        accs.clear()
    #
    for i in range(len(cls)):
        ious.append(iou[cls[i]])
        plt.figure()
        plt.hist(ious, label=cls[i] + '_IoU')
        plt.legend()
        plt.savefig('/storage/sjpark/vehicle_data/histogram/FCN/FCN8s/256/{}/{}'.format(cls[i], cls[i] + '_IoUS'))
        plt.close()
        ious.clear()

    # for i in range(len(cls)):
    #     precisions.append(precision[cls[i]])
    #     plt.figure()
    #     plt.hist(precisions, label=cls[i] + '_Precision')
    #     plt.legend()
    #     plt.savefig('/storage/sjpark/vehicle_data/histogram/FCN/FCN8s/512/{}/{}'.format(cls[i], cls[i] + '_Precision'))
    #     plt.close()
    #     precisions.clear()
    # #
    # for i in range(len(cls)):
    #     recalls.append(recall[cls[i]])
    #     plt.figure()
    #     plt.hist(recalls, label=cls[i] + '_Recall')
    #     plt.legend()
    #     plt.savefig('/storage/sjpark/vehicle_data/histogram/FCN/FCN8s/512/{}/{}'.format(cls[i], cls[i] + '_Recall'))
    #     plt.close()
    #     recalls.clear()











