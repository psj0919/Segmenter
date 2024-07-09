import torch
import os
import argparse
import tqdm
import json
import numpy as np

from copy import deepcopy
from dataset.dataset import vehicledata
from Segmenter.factory import create_segmenter
from optim.factory import create_optimizer, create_scheduler
from timm import scheduler
from timm import optim
from optim.scheduler import PolynomiaLR
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_cfg, self.decoder_cfg = self.setup_model_config(self.cfg)
        self.device = self.setup_device()
        self.train_loader = self.get_dataloader()
        self.val_loader = self.get_val_dataloader()
        self.opt_cfg = self.setup_opt_cfg()
        self.model = self.setup_network()
        # self.optimizer = self.setup_optimizer(self.opt_cfg, self.model)
        # self.scheduler = self.setup_scheduler(self.opt_cfg, self.optimizer)
        self.optimizer = self.setup_optimizer_adam()
        self.scheduler = self.setup_scheduler_step()
        self.loss = self.setup_loss()
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

    def setup_opt_cfg(self):
        optimizer_kwargs = dict(
            opt = 'adam',
            sched= "polynomial" ,
            lr=self.cfg['dataset']['lr'],
            weight_decay=self.cfg['solver']['weight_decay'],
            momentum=0.9,
            clip_grad=None,
            epochs=self.cfg['dataset']['epochs'],
            min_lr=1e-5,
            poly_power=0.9,
            poly_step_size=1,

        )
        optimizer_kwargs['iter_max'] = len(self.train_loader) * optimizer_kwargs['epochs']
        optimizer_kwargs['iter_warmup'] = 0.0
        opt_args = argparse.Namespace()
        opt_vars = vars(opt_args)
        for k, v in optimizer_kwargs.items():
            opt_vars[k] = v

        return opt_args

    def get_dataloader(self):
        if self.cfg['dataset']['name'] == 'vehicledata':
            dataset = vehicledata(self.cfg['dataset']['img_path'], self.cfg['dataset']['ann_path'],
                                  self.cfg['dataset']['num_class'])
        else:
            raise ValueError("Invalid dataset name...")
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg['dataset']['batch_size'], shuffle=True,
                                             num_workers=self.cfg['args']['num_workers'])

        return loader

    def get_val_dataloader(self):
        if self.cfg['dataset']['name'] == 'vehicledata':
            val = vehicledata(self.cfg['dataset']['val_path'], self.cfg['dataset']['val_ann_path'],
                                      self.cfg['dataset']['num_class'])
        else:
            raise ValueError("Invalid dataset name..")

        loader = torch.utils.data.DataLoader(val, batch_size = 1, shuffle = False, num_workers = self.cfg['args']['num_workers'])

        return loader

    def setup_optimizer_adam(self):
        if self.cfg['solver']['optimizer'] == "sgd":
            optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.cfg['solver']['lr'],
                                        weight_decay=self.cfg['solver']['weight_decay'])
        elif self.cfg['solver']['optimizer'] == "adam":
            optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.cfg['solver']['lr'],
                                         weight_decay=self.cfg['solver']['weight_decay'])
        else:
            raise NotImplementedError("Not Implemented {}".format(self.cfg['solver']['optimizer']))

        return optimizer

    def setup_scheduler_step(self):
        if self.cfg['solver']['scheduler'] == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.cfg['solver']['step_size'],
                                                        self.cfg['solver']['gamma'])
        elif self.cfg['solver']['scheduler'] == 'cyclelr':
            scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-7,
                                                          max_lr=self.cfg['solver']['lr'],
                                                          step_size_up=int(self.cfg['dataset']['epochs'] / 5 * 0.7),
                                                          step_size_down=int(self.cfg['dataset']['epochs'] / 5) - int(self.cfg['dataset']['epochs'] / 5 * 0.7),
                                                          cycle_momentum=False,
                                                          gamma=0.9)

        return scheduler

    def setup_network(self):
        model = create_segmenter(self.cfg)

        return model.to(self.device)

    def setup_optimizer(self, opt_args, model):
        return optim.create_optimizer(opt_args, model)

    def setup_scheduler(self, opt_args, optimizer):
        return create_scheduler(opt_args, optimizer)

    def setup_loss(self):
        if self.cfg['solver']['loss'] == 'crossentropy':
            loss = torch.nn.CrossEntropyLoss(reduction='sum')
        else:
            raise("Please check loss name...")
        return loss


    def training(self):
        print("-----strat training-----")
        self.model.train()
        num_updates = self.cfg['dataset']['epochs'] * len(self.train_loader)

        for curr_epoch in range(self.cfg['dataset']['epochs']):
            if (curr_epoch + 1) % 3 == 0:
                avr_ious, pixel_accs, cls, org_cls, target_crop_image, pred_crop_image = self.validation()
                # Iou
                for i in range(len(avr_ious)):
                    self.writer.add_scalar(tag='total_ious/{}'.format(cls[i]), scalar_value=avr_ious[i], global_step = self.global_step)
                # Crop Image
                for i in range(len(target_crop_image)):
                    self.writer.add_image('target /' + org_cls[i], self.trg_to_class_rgb(target_crop_image[i], org_cls[i]),
                                          dataformats='HWC', global_step=1)
                    self.writer.add_image('pred /' + org_cls[i], self.pred_to_class_rgb(pred_crop_image[i], org_cls[i]),
                                          dataformats='HWC', global_step=1)
                # Pixel Acc
                self.writer.add_scalar(tag='pixel_accs', scalar_value=pixel_accs.mean(), global_step=self.global_step)
            #
            for batch_idx, (data, target, label, index) in (enumerate(self.train_loader)):

                self.global_step += 1
                data = data.to(self.device)
                target = target.to(self.device)
                label = label.to(self.device)
                #
                label = label.type(torch.long)
                out = self.model.forward(data)
                out = out.type(torch.float32)
                #
                loss = 10 * self.loss(out, label)
                #
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #

                #
                if self.global_step % self.cfg['solver']['print_freq'] == 0:
                    self.writer.add_scalar(tag='train/loss', scalar_value=loss, global_step=self.global_step)
                if self.global_step % (10 * self.cfg['solver']['print_freq']) ==0:
                    self.writer.add_image('train/train_image', self.matplotlib_imshow(data[0].to('cpu')),
                                          dataformats='HWC', global_step=self.global_step)
                    self.writer.add_image('train/predict_image',
                                          self.pred_to_rgb(out[0]),
                                          dataformats='HWC', global_step=self.global_step)
                    self.writer.add_image('train/target_image',
                                          self.trg_to_rgb(target[0]),
                                          dataformats='HWC', global_step=self.global_step)
            print("Complete {}_epoch".format(curr_epoch))

            # num_updates += 1
            # self.scheduler.step_update(num_updates=num_updates)
            self.scheduler.step()

            self.writer.add_scalar(tag='train/lr', scalar_value=self.optimizer.param_groups[0]['lr'],
                                   global_step=curr_epoch)

            if self.global_step % 2 == 0:
                self.save_model(self.cfg['model']['checkpoint'])

    def validation(self):
        self.model.eval()
        total_ious = []
        total_accs = []
        cls = []
        cls_count = []

        for iter, (data, target, label, idx) in enumerate(self.val_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            logits = self.model(data)
            pred = logits.softmax(dim=1).argmax(dim=1).to('cpu')
            target_ = target.softmax(dim=1).argmax(dim=1).to('cpu')
            file, json_path = self.load_json_file(int(idx))
            # Iou
            iou = self.make_bbox(file, json_path, target_, pred)
            # Crop image
            target_crop_image, pred_crop_image, org_cls = self.crop_image(target[0], logits[0], json_path)

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
                total_accs.append(self.pixel_acc(p.cpu(), t.cpu()))

        pixel_accs = np.array(total_accs).mean()

        self.model.train()

        return avr_ious, pixel_accs, cls, org_cls, target_crop_image, pred_crop_image



    def save_model(self, save_path):
        save_file = 'Segmenter_epochs:{}_optimizer:{}_lr:{}_model{}.pth'.format(self.cfg['dataset']['epochs'],
                                                                          self.cfg['solver']['optimizer'],
                                                                          self.cfg['dataset']['lr'],
                                                                          self.cfg['model']['backbone']['name'])
        path = os.path.join(save_path, save_file)
        torch.save({'model': deepcopy(self.model), 'optimizer': self.optimizer.state_dict()}, path)
        print("Success save")

    def load_json_file(self, idx):
        json_path = '/storage/sjpark/vehicle_data/Dataset/val_json/'
        idx_json = sorted(os.listdir(json_path))
        expected_cls = ['background', 'motorcycle', 'bicycle', 'twowheeler', 'pedestrian', 'rider', 'sidewalk',
                        'crosswalk', 'speedbump', 'redlane', 'stoplane', 'trafficlight']
        for i in range(len(expected_cls)):
            expected_cls[i] = expected_cls[i].lower()

        json_file = os.path.join(json_path, idx_json[idx])
        with open(json_file, 'r') as f:
            json_data1 = json.load(f)
        json_data = json_data1['annotations']
        # json_cls_data = json_data1['class']
        ann_json = []
        #
        for i in range(len(json_data)):
            if any(keyword in json_data[i]['class'] for keyword in expected_cls):
                pass
            else:
                ann_json.append(json_data[i])

        return idx_json[idx], ann_json

    def make_bbox(self, file, json_path, target_image, pred_image):
        ious = []
        org_cls = []
        #
        org_res = (1920, 1080)
        target_res = (224, 224)
        #
        scale_x = target_res[0] / org_res[0]
        scale_y = target_res[1] / org_res[1]
        count = 0
        for i in range(len(json_path)):
            polygon = json_path[i]['polygon']
            cls = json_path[i]['class']
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
                if (x_min == x_max) or (y_min==y_max):
                    pass
                else:
                    #
                    crop_target_image = target_image[:, y_min:y_max:, x_min:x_max]
                    crop_pred_image = pred_image[:, y_min:y_max:, x_min:x_max]
                    #
                    crop_target_image = torch.where(crop_target_image >= 1, torch.tensor(1.0), torch.tensor(0.0))
                    crop_pred_image = torch.where(crop_pred_image >= 1, torch.tensor(1.0), torch.tensor(0.0))
                    org_cls.append(cls)
                    iou = self.iou(crop_pred_image, crop_target_image, cls)
                    ious.append(iou)

        return ious

    def iou(self, pred, target, cls, thr=0.5, dim=(2, 3), epsilon=0.001):
        ious = {}
        y_true = target.to(torch.float32)
        y_pred = pred.to(torch.float32)

        inter = (y_true * y_pred).sum(dim=(0, 1))
        union = (y_true + y_pred - y_true * y_pred).sum(dim=(0, 1))

        iou = ((inter + epsilon) / (union + epsilon)).mean()
        ious[cls] = iou
        return ious

    def crop_image(self, target, pred, json_path):
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
                if x_max > 512:
                    x_max = 512
                #
                y_max = np.max(polygon[:, 1]) + 20
                if y_max > 512:
                    y_max = 512
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

    def pred_to_rgb(self, pred):
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
        for i in range(self.cfg['dataset']['num_class']):
            pred_rgb[pred == i] = np.array(color_table[i])

        return pred_rgb



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

    def trg_to_class_rgb(self, target, cls):
        assert len(target.shape) == 3

        CLASSES = [
            'background', 'vehicle', 'bus', 'truck', 'policeCar', 'ambulance', 'schoolBus', 'otherCar',
            'freespace', 'curb', 'safetyZone', 'roadMark', 'whiteLane',
            'yellowLane', 'blueLane', 'constructionGuide', 'trafficDrum',
            'rubberCone', 'trafficSign', 'warningTriangle', 'fence'
        ]
        for i in range(len(CLASSES)):
            CLASSES[i] = CLASSES[i].lower()

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

    def pred_to_class_rgb(self, pred, cls):
        assert len(pred.shape) == 3
        #
        CLASSES = [
            'background', 'vehicle', 'bus', 'truck', 'policeCar', 'ambulance', 'schoolBus', 'otherCar',
            'freespace', 'curb', 'safetyZone', 'roadMark', 'whiteLane',
            'yellowLane', 'blueLane', 'constructionGuide', 'trafficDrum',
            'rubberCone', 'trafficSign', 'warningTriangle', 'fence'
        ]
        for i in range(len(CLASSES)):
            CLASSES[i] = CLASSES[i].lower()
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
        i = CLASSES.index(cls.lower())
        pred_rgb[pred == i] = np.array(color_table[i])

        return pred_rgb

    @staticmethod
    def matplotlib_imshow(img):
        assert len(img.shape) == 3
        npimg = img.numpy()
        return (np.transpose(npimg, (1, 2, 0))[:, :, ::-1] * 255).astype(np.uint8)

    @staticmethod
    def pixel_acc(pred, target):
        correct = (pred ==target).sum()
        total = (target== target).sum()

        return correct / total





