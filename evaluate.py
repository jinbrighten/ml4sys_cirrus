import sys
sys.path.append('./')

from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from typing import Any
import os
import pickle
import torch
import json

import numpy as np
from tqdm import tqdm

from eval_visualize.kitti_visualize import cnf, process_sample
from eval_visualize.iou_box3d import box3d_overlap

CLASS_NAME_TO_ID = {
    'Pedestrian': 0,
    'Car': 1,
    'Cyclist': 2,
    'Van': 1,
    'Truck': -3,
    'Person_sitting': 0,
    'Tram': -99,
    'Misc': -99,
    'DontCare': -1
}

LABEL_PER_MODEL = {
    "3dssd": 0,
    "pointpillars": 2,
    "pv_rcnn": 2,
    "parta2": 0
}

draw_half = True

if draw_half:
    # '''
    boundary = {
        "minX": 0,
        "maxX": 76,
        "minY": -38,
        "maxY": 38,
        "minZ": -2.73,
        "maxZ": 1.27
    }
    # '''
    '''
    boundary = {
        "minX": 0,
        "maxX": 100,
        "minY": -50,
        "maxY": 50,
        "minZ": -10,
        "maxZ": 10
    }
    '''
else:
    boundary = {
        "minX": -50,
        "maxX": 50,
        "minY": -50,
        "maxY": 50,
        "minZ": -2.73,
        "maxZ": 1.27
    }



def get_label(label_path):
    labels = []
    for line in open(label_path, 'r'):
        line = line.rstrip()
        line_parts = line.split(' ')

        obj_name = line_parts[0]  # 'Car', 'Pedestrian', ...
        cat_id = int(CLASS_NAME_TO_ID[obj_name])
        if cat_id != 1:  # ignore Tram and Misc
            continue
        truncated = int(float(line_parts[1]))  # truncated pixel ratio [0..1]
        occluded = int(line_parts[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        alpha = float(line_parts[3])  # object observation angle [-pi..pi]
        # xmin, ymin, xmax, ymax
        bbox = np.array([float(line_parts[4]), float(line_parts[5]), float(line_parts[6]), float(line_parts[7])])
        # height, width, length (h, w, l)
        h, w, l = float(line_parts[8]), float(line_parts[9]), float(line_parts[10])
        # location (x,y,z) in camera coord.
        x, y, z = float(line_parts[11]), float(line_parts[12]), float(line_parts[13])
        ry = float(line_parts[14])  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

        cat_id = 0
        object_label = [cat_id, x, y, z, h, w, l, ry]
        labels.append(object_label)

    if len(labels) == 0:
        labels = np.zeros((1, 8), dtype=np.float32)
        has_labels = False
    else:
        labels = np.array(labels, dtype=np.float32)
        has_labels = True

    return labels, has_labels

def center_to_corner_box3d(boxes_center):
    # (N, 7) -> (N, 8, 3)
    N = boxes_center.shape[0]
    ret = np.zeros((N, 8, 3), dtype=np.float32)

    for i in range(N):
        box = boxes_center[i]
        translation = box[0:3]
        size = box[3:6]
        rotation = [0, 0, box[-1]]

        h, w, l = size[0], size[1], size[2]
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0, 0, 0, 0, h, h, h, h]])

        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
        box3d = cornerPosInVelo.transpose()
        ret[i] = box3d

    return ret

def box3d_convert(input_box):
    # Convert mmdet3d style bbox to py3d style bbox
    output_box = np.zeros_like(input_box, dtype=np.float32)

    output_box[:,0,:] = input_box[:,1,:]
    output_box[:,1,:] = input_box[:,0,:]
    output_box[:,2,:] = input_box[:,3,:]
    output_box[:,3,:] = input_box[:,2,:]
    output_box[:,4,:] = input_box[:,5,:]
    output_box[:,5,:] = input_box[:,4,:]
    output_box[:,6,:] = input_box[:,7,:]
    output_box[:,7,:] = input_box[:,6,:]
    return output_box

def get_filtered_bbox(boundary, labels):
    minX = boundary['minX']
    maxX = boundary['maxX']
    minY = boundary['minY']
    maxY = boundary['maxY']
    minZ = boundary['minZ']
    maxZ = boundary['maxZ']


    label_x = (labels[:, 0] >= minX) & (labels[:, 0] < maxX)
    label_y = (labels[:, 1] >= minY) & (labels[:, 1] < maxY)
    label_z = (labels[:, 2] >= minZ) & (labels[:, 2] < maxZ)
    mask_label = label_x & label_y & label_z
    labels = labels[mask_label]
    return labels


def evaluate(sampled_path: str, output_path: str, model: str, idx: str) -> None:
    raw_root = "/data/3d/kitti/training/"
    sampled_root = "/data/3d/kitti_sampled/training/pre_infer/multi_resolution/"
    output_root = f"/data/3d/kitti_sampled/training/post_infer/{model}/multi_resolution/"
    result_root = f"/data/3d/kitti_sampled/training/results/{model}/multi_resolution/"

    # sampled_path = os.path.join(sampled_root, input_args)
    # output_path = os.path.join(output_root, input_args)
    # result_path = os.path.join(result_root, input_args)
    # sampled_path = f"/data/3d/mlfs/pre_infer/{idx}"
    # output_path = f"/data/3d/mlfs/post_infer/{idx}"

    print("=====================================")
    print(f"Start evaluation: {model}")
    print("=====================================")

    F1_scores = []
    validate_indices = []

    GT = 0
    FN = 0
    FP = 0
    TP = 0
    space_savings = []
    for sample_idx in range(100):
        if not os.path.isfile(os.path.join(sampled_path, "%06d.bin"%sample_idx)) or not os.path.isfile(os.path.join(output_path, "%06d.pkl"%sample_idx)):
            print(f"Index {sample_idx} is not existed")
            continue
        output_file = os.path.join(output_path, "%06d.pkl"%sample_idx)    

        GT_file = os.path.join(raw_root, "label_2/%06d.txt"%sample_idx)
        GT_pc = os.path.join(raw_root, "velodyne/%06d.bin"%sample_idx)
        sample_pc = os.path.join(sampled_path, "%06d.bin"%sample_idx)

        GT_pc_size = np.fromfile(GT_pc, dtype=np.float32).reshape(-1, 4).shape[0]
        sample_pc_size = np.fromfile(sample_pc, dtype=np.float32).reshape(-1, 4).shape[0]
        space_savings.append(1 - sample_pc_size / GT_pc_size)

        if not os.path.exists(GT_file):
            continue

        validate_indices.append(int(sample_idx))

        _, _, _, _, _, labels_lidar = process_sample(raw_root, sample_idx)

        GT_mask = labels_lidar[:, 0] == 1
        labels_car = labels_lidar[GT_mask]

        if labels_car.size != 0:
            GT += np.count_nonzero(GT_mask)

            with open(output_file, 'rb') as f:
                inference_result = pickle.load(f)
            try:
                boxes = inference_result["boxes"]
                scores = inference_result["scores"]
                labels = inference_result["labels"]

                label_mask = labels == LABEL_PER_MODEL[model]
                boxes = boxes[label_mask]
                scores = scores[label_mask]
                labels = labels[label_mask]

                score_mask = scores > 0.5

                boxes = boxes[score_mask]
                scores = scores[score_mask]
                labels = labels[score_mask]

                boxes = boxes[:, [0,1,2,5,4,3,6]]

                if len(boxes) == 0:
                    FN += np.count_nonzero(GT_mask)
                else:
                    boxes = get_filtered_bbox(boundary, boxes)

                    GT_corners = center_to_corner_box3d(labels_car[:, 1:])
                    boxes_corners = center_to_corner_box3d(boxes)

                    GT_bbox, boxes_bbox = box3d_convert(GT_corners), box3d_convert(boxes_corners)

                    GT_tensor = torch.from_numpy(GT_bbox)
                    boxes_tensor = torch.from_numpy(boxes_bbox)

                    intersection_vol, iou_3d_raw, miou_raw = box3d_overlap(GT_tensor, boxes_tensor)

                    FN += iou_3d_raw.shape[0] - torch.nonzero(iou_3d_raw).shape[0]
                    FP += iou_3d_raw.shape[1] - torch.nonzero(iou_3d_raw).shape[0]
                    TP += torch.nonzero(iou_3d_raw).shape[0]

            except Exception as e:
                print(f"Index {sample_idx} is not existed")
                continue
            
            try:
                precision = TP/(TP+FP)
            except:
                precision = 0
            
            try:
                recall = TP/(TP+FN)
            except:
                recall = 0

            try:
                f1_score = 2 * (precision * recall) / (precision + recall)
            except:
                f1_score = 0
            F1_scores.append(float(f1_score))

    average_F1 = np.mean(F1_scores)
    average_space_saving = np.mean(space_savings)
    
    return average_F1, average_space_saving
        
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        
    with open(os.path.join(result_path, "evaluation.json"), 'w') as fp:
        json.dump({"F1": average_F1, "Space saving": average_space_saving}, fp)

def main(args: Namespace) -> None:
    mlfs_root = f"/data/3d/mlfs/{args.name}/"
    while True:
        for idx in os.listdir(os.path.join(mlfs_root, "flag/pre_eval/")):
            os.remove(os.path.join(mlfs_root, f"flag/pre_eval/{idx}"))
            f1, space_saving = evaluate(
                os.path.join(mlfs_root, f"pre_infer/{idx}"),
                os.path.join(mlfs_root, f"post_infer/{idx}"),
                args.model,
                idx
            )
            with open(os.path.join(mlfs_root, f"post_eval/{idx}"), 'w') as fp:
                json.dump({"f1": f1, "saving": space_saving}, fp)
            with open(os.path.join(mlfs_root, f"flag/post_eval/{idx}"), 'w') as fp:
                fp.write("1")

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("--model", type=str, default="3dssd")
    parser.add_argument("-a", "--args", nargs="*", type=int)
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.set_defaults(func=main)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    args.func(args)