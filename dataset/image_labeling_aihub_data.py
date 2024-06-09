from preprocessing.mmsegmentation.color_param import COLOR_PARAM
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import json
import traceback
from tqdm.auto import tqdm
import datetime
import argparse

CP = COLOR_PARAM()
error_count = 0
error_path = []

except_classes = ['motorcycle', 'bicycle', 'twowheeler', 'pedestrian', 'rider', 'sidewalk', 'crosswalk', 'speedbump', 'redlane', 'stoplane', 'trafficlight']

def parse_args():
    parser = argparse.ArgumentParser(
        description='make annotation image')
    parser.add_argument('data_root', help='set data root')
    args = parser.parse_args()
    
    return args

def print_nowtime():
    now = datetime.datetime.now()
    return now

def get_dir_list(path):
    root_dir = []
    for file_path in listdir(path):
        path_str = os.path.join(path, file_path)
        root_dir.append(path_str)
    return root_dir

def read_json(json_path):
    with open(json_path) as json_file:
        dict_json = json.load(json_file)

        return dict_json

def make_anno_img(json, anno_file_path,json_file_path, overlap_file_path=None):
    global error_path
    
    width = json.get('information').get('resolution')[0]
    height = json.get('information').get('resolution')[1]
    
    annotations = json.get('annotations')
    b_count = 0
    for i in range(len(annotations)):
        if annotations[i]['class'].lower() != 'background':
            break
        else:
            b_count += 1
            
    if annotations == []:
        return 0
    
    img = Image.new('L', (width, height), 0)
    merged_mask = None
    for idx, anno in enumerate(annotations):
        polygons = anno.get('polygon')
        class_name = anno.get('class')
        if class_name.lower() in except_classes:
            continue
    
        color_idx = CP.get_class_lowercase().index(class_name.lower())
        color = CP.PALETTE[color_idx]

        try:
            ImageDraw.Draw(img).polygon(polygons, fill=int(color_idx))
        except:
            # print(anno_file_path)
            # print(class_name)
            # print(polygons)
            # print(traceback.format_exc())
            error_path.append(json_file_path)
            
        polygon = np.array(img)

        if merged_mask is None:
            merged_mask = polygon
        else:
            merged_mask += polygon
            
    mask_image = Image.fromarray(merged_mask, "L")
    if overlap_file_path:
        mask_image.save(os.path.join(overlap_file_path))
    img.save(os.path.join(anno_file_path))
    
    

if __name__ == '__main__':
    args = parse_args()
    data_root = args.data_root
    CP = COLOR_PARAM()
    
    error_count = 0
    data_path = data_root
    
    now = print_nowtime()
    print(f'[{now}] Start Make annotation images...')
    
    json_path = os.path.join(data_path,'val_label')
    anno_path = os.path.join(data_path, 'ann_val')
    # make file list
    x = os.listdir(json_path)
    files = []
    for i in x:
        json_path = os.path.join(json_path, i+'/sensor_raw_data/camera')
        for (a, b, c) in os.walk(json_path):
            if c:
                for file in c:
                    file_name = os.path.splitext(file)[0]
                    json_file_path = os.path.join(a, file)
                    anno_file_path = os.path.join(anno_path, f"{file_name}.png")
                    make_anno_img(read_json(json_file_path), anno_file_path, json_file_path)
                    json_path = '/storage/sjpark/vehicle_data/Dataset/val_label/'

    f_now = print_nowtime()
    print(f'[{f_now}] Finish!, Work time: {f_now-now}')
    print(f'Error path : {error_path}')





        




