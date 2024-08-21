import torch
import os
import argparse
import tqdm
import json
import numpy as np

from core.functions import *
from copy import deepcopy
from dataset.dataset import vehicledata
from Segmenter.factory import create_segmenter
from optim.factory import create_optimizer, create_scheduler
from timm import scheduler
from timm import optim
from optim.scheduler import PolynomiaLR
from torch.utils.tensorboard import SummaryWriter

except_classes = ['motorcycle', 'bicycle', 'twowheeler', 'pedestrian', 'rider', 'sidewalk', 'crosswalk', 'speedbump', 'redlane', 'stoplane', 'trafficlight']

CLASSES = [
    'background', 'vehicle', 'bus', 'truck', 'policeCar', 'ambulance', 'schoolBus', 'otherCar',
    'freespace', 'curb', 'safetyZone', 'roadMark', 'whiteLane',
    'yellowLane', 'blueLane', 'constructionGuide', 'trafficDrum',
    'rubberCone', 'trafficSign', 'warningTriangle', 'fence'
]

class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_cfg, self.decoder_cfg = self.setup_model_config(self.cfg)
        self.device = self.setup_device()
        self.test_loader = self.get_test_dataloader()
        self.model = self.setup_network()
        self.save_path = self.cfg['model']['save_dir']
        self.writer = SummaryWriter(log_dir = self.save_path)
        self.global_step = 0


    def setup_device(self):
        if self.cfg['args']['gpu_id'] is not None:
            device = torch.device("cuda:{}".format(self.cfg['args']['gpu_id']) if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        return device
    def setup_model_config(self,cfg):
        model_cfg = self.cfg["model"]["backbone"]
        decoder_cfg = self.cfg["decoder"]

        return model_cfg, decoder_cfg

    def get_test_dataloader(self):
        if self.cfg['dataset']['name'] == 'vehicledata':
            dataset = vehicledata(self.cfg['dataset']['img_path'], self.cfg['dataset']['ann_path'],
                                  self.cfg['dataset']['num_class'])
        else:
            raise ValueError("Invalid dataset name...")
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg['dataset']['batch_size'], shuffle=True,
                                             num_workers=self.cfg['args']['num_workers'])

        return loader

    def setup_network(self):
        model = create_segmenter(self.cfg)

        return model.to(self.device)

    def test(self):
        self.model.eval()
        total_ious = []
        total_accs = {}
        cls = []
        cls_count = []
        #
        for i in range(len(CLASSES)):
            CLASSES[i] = CLASSES[i].lower()
        for c in CLASSES:
            if c in except_classes:
                pass
            else:
                total_accs[c] = []
        #
        for curr_epoch in range(self.cfg['args']['epochs']):
            for iter, (data, target, label, idx) in enumerate(self.test_loader):
                self.global_step += 1
                #
                data = data.to(self.device)
                target = target.to(self.device)
                label = label.to(self.device)

                logits = self.model(data)
                pred = logits.softmax(dim=1).argmax(dim=1).to('cpu')
                pred_ = pred.to(self.device)
                pred_softmax = logits.softmax(dim=1)
                target_ = target.softmax(dim=1).argmax(dim=1).to('cpu')
                file, json_path = load_json_file(int(idx))
                # Iou
                iou = make_bbox(file, json_path, target_, pred)
                # Crop image
                target_crop_image, pred_crop_image, org_cls = crop_image(target[0], logits[0], json_path)

                for i in range(len(iou)):
                    for key, val in iou[i].items():
                        if key in cls:
                            a = cls.index(key)
                            total_ious[a] += val
                            cls_count[a] += 1
                        else:
                            cls.append(key)
                            total_ious.append(val)
                            cls_count.append(1)

                avr_ious = [total / count for total, count in zip(total_ious, cls_count)]

                # Pixel Acc
                for p, t in zip(pred, label):
                    x = pixel_acc_cls(p.cpu(), t.cpu(), cls)
                #
                for idx, c in enumerate(cls):
                    total_accs[c].append(x[idx])
                #
                precision, recall = precision_recall(label, pred_softmax[0], threshold = 0.5)

                if self.global_step % 3 == 0:
                    #
                    for i in range(len(avr_ious)):
                        self.writer.add_scalar(tag='total_ious/{}'.format(cls[i]), scalar_value=avr_ious[i], global_step = self.global_step)

                    # Crop Image
                    for i in range(len(target_crop_image)):
                        self.writer.add_image('target /' + org_cls[i], trg_to_class_rgb(target_crop_image[i], org_cls[i]),
                                              dataformats='HWC', global_step=self.global_step)
                        self.writer.add_image('pred /' + org_cls[i], pred_to_class_rgb(pred_crop_image[i], org_cls[i]),
                                              dataformats='HWC', global_step=self.global_step)
                    # Pixel Acc
                    for i in range(len(cls)):
                        self.writer.add_scalar(tag='pixel_accs/{}'.format(cls[i]), scalar_value=np.mean(total_accs[cls[i]]), global_step=self.global_step)

                    # precision & recall
                    for i in range(len(CLASSES)):
                        self.writer.add_scalar(tag='precision/{}'.format(CLASSES[i]), scalar_value=precision[CLASSES[i]], global_step=self.global_step)
                    for i in range(len(cls)):
                        self.writer.add_scalar(tag='recall/{}'.format(CLASSES[i]), scalar_value=recall[CLASSES[i]], global_step=self.global_step)

                    #
                    self.writer.add_image('train/predict_image',
                                          pred_to_rgb(logits[0]),
                                          dataformats='HWC', global_step=self.global_step)
                    #
                    self.writer.add_image('train/target_image',
                                          trg_to_rgb(target[0]),
                                          dataformats='HWC', global_step=self.global_step)









