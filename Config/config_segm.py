import yaml
from pathlib import Path

import os

def dataset_info(dataset_name='vehicledata'):
    if dataset_name == 'vehicledata':
        train_path = "/storage/sjpark/vehicle_data/Dataset/train_image/"
        ann_path = "/storage/sjpark/vehicle_data/Dataset/ann_train/"
        val_path = '/storage/sjpark/vehicle_data/Dataset/val_image/'
        val_ann_path = '/storage/sjpark/vehicle_data/Dataset/ann_val/'
        test_path = '/storage/sjpark/vehicle_data/Dataset/test_image/'
        test_ann_path = '/storage/sjpark/vehicle_data/Dataset/ann_test/'
        json_file = '/storage/sjpark/vehicle_data/Dataset/val_json/'
        num_class = 21
    else:
        raise NotImplementedError("Not Implemented dataset name")

    return dataset_name, train_path, ann_path, val_path, val_ann_path, test_path, test_ann_path, json_file, num_class

def get_config_dict():
    dataset_name = "vehicledata"
    name, img_path, ann_path, val_path, val_ann_path, test_path, test_ann_path, json_file, num_class, = dataset_info(dataset_name)

    dataset = dict(
        name = name,
        img_path = img_path,
        ann_path = ann_path,
        val_path = val_path,
        val_ann_path = val_ann_path,
        test_path = test_path,
        test_ann_path = test_ann_path,
        json_file = json_file,
        num_class = num_class,
        epochs = 200,
        eval_freq =2,
        batch_size = 8,
        image_size= 512,
        crop_size= 512,
        window_stride= 512
    )
    args = dict(
        gpu_id='0',
        num_workers = 3
    )
    solver = dict(
        optimizer = "adam",
        scheduler='cyclelr',
        step_size=5,
        gamma=0.95,
        loss="crossentropy",
        weight_decay=5e-4,
        print_freq=20,
        lr=1e-4,

    )
    model = dict(
        backbone = dict(
            name= " vit_base_patch8_384",
            image_size= (256, 256),
            patch_size= 8,
            d_model= 768,
            n_heads= 8,
            n_layers= 6,
            d_ff = 0,
            normalization= "vit",
            n_cls= num_class,
            distilled= False,
        ),
        resume = '',
        save_dir = '/storage/sjpark/vehicle_data/runs/Segmenter/train/256',
        checkpoint = '/storage/sjpark/vehicle_data/checkpoints/segm/256'
    )

    decoder = dict(
        name = "mask_transformer",
        drop_path_rate = 0.0,
        dropout = 0.1,
        n_layers = 2,
        n_cls = num_class,

    )

    config = dict(
        args= args,
        dataset = dataset,
        model = model,
        solver = solver,
        decoder = decoder,
    )

    return config



