from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from typing import Any
import os
import json
import pickle

from mmdet3d.apis import init_model, inference_detector
import numpy as np
from tqdm import tqdm

MODEL_CONFIG = {
    # "3dssd": "../lib/mmdetection3d/configs/3dssd/3dssd_4xb4_kitti-3d-car.py", # 16384
    # "pointpillars": "../lib/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py",
    # "pv_rcnn": "../lib/mmdetection3d/configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py",
    # "parta2": "../lib/mmdetection3d/configs/parta2/parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-car.py",
    # "point_rcnn": "../lib/mmdetection3d/configs/point_rcnn/point-rcnn_8xb2_kitti-3d-3class.py"
    "3dssd": "/workspaces/ml4sys_cirrus/mmdetection3d/configs/3dssd/3dssd_4xb4_kitti-3d-car.py", # 16384
    "pointpillars": "/workspaces/ml4sys_cirrus/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py",
    "pv_rcnn": "/workspaces/ml4sys_cirrus/mmdetection3d/configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py",
    "parta2": "/workspaces/ml4sys_cirrus/mmdetection3d/configs/parta2/parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-car.py",
    "point_rcnn": "/workspaces/ml4sys_cirrus/mmdetection3d/configs/point_rcnn/point-rcnn_8xb2_kitti-3d-3class.py"
}

MODEL_CHECKPOINT = {
    "3dssd": "3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth",
    "pointpillars": "hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth",
    "pv_rcnn": "pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth",
    "parta2": "hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car_20210831_022017-cb7ff621.pth",
    "point_rcnn": "point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth"
}

raw_path = "/data/3d/kitti/training/"

def inference(points: np.ndarray, model: Any):
    result = inference_detector(model, points)[0]
    return result


def main(args: Namespace) -> None:
    input_root = "/data/3d/kitti_sampled/training/pre_infer/ml4sys/"
    output_root = "/data/3d/kitti_sampled/training/post_infer/ml4sys/"

    config_file = MODEL_CONFIG[args.model]
    checkpoint_file = os.path.join("/data/3d/mmdet3d_checkpoints", MODEL_CHECKPOINT[args.model])

    input_path = input_root
    output_path = os.path.join(output_root, args.model)
    
    if args.args:
        suffix = "_".join(map(str, args.args)) + "_"
        input_path = os.path.join(input_path, suffix)
        output_path = os.path.join(output_path, suffix)
    
    input_path = os.path.join(input_path, "")
    output_path = os.path.join(output_path, "")
    
    print("=====================================")
    print(f"Start inference: {args.model}")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print("=====================================")
    inference_directory(args.model, args.input_root, args.output_root, args.device)
    

def inference_directory(model_name: str, input_dir: str, output_dir: str, device) -> None:
    config_file = MODEL_CONFIG[model_name]
    checkpoint_file = os.path.join("/data/3d/mmdet3d_checkpoints", MODEL_CHECKPOINT[model_name])

    # input_dir = args.input_root
    # output_dir = args.output_root
    os.makedirs(output_dir, exist_ok=True)
    
    device = f"cuda:{device}"
    model = init_model(config_file, checkpoint_file, device)
    if os.path.isdir(input_path):
        for binfile in tqdm(os.listdir(input_path), desc=output_path):
            try:
                if not binfile.endswith('.bin'): continue
                points = np.fromfile(os.path.join(input_path, binfile), dtype=np.float32).reshape(-1, 4)
                result = inference(points, model)

                boxes = result.pred_instances_3d.bboxes_3d
                scores = result.pred_instances_3d.scores_3d
                labels = result.pred_instances_3d.labels_3d

                inference_result = dict()
                inference_result["boxes"] = boxes.cpu().numpy()
                inference_result["scores"] = scores.cpu().numpy()
                inference_result["labels"] = labels.cpu().numpy()
                
            except Exception as e:
                print(f"Error: {e}")
                continue

            with open(os.path.join(output_path, binfile[:-3]+'pkl'), 'wb') as fp:
                pickle.dump(inference_result, fp)



def get_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("--device", type=int, default="4")
    parser.add_argument("--model", type=str, default="pv_rcnn")
    parser.add_argument("-i", "--input_root", type=str, default="/data/3d/kitti_sampled/training/pre_infer/")
    parser.add_argument("-o", "--output_root", type=str, default="/data/3d/kitti_sampled/training/post_infer/")
    parser.set_defaults(func=main)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    args.func(args)