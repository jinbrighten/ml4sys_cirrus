import linecache
import multiprocessing
import os
import pickle
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_visualize.kitti_visualize import cnf, process_sample
from eval_visualize.iou_box3d import box3d_overlap



def main(args: Namespace) -> None:
    gt_root = "/data/3d/waymo_v1/validation/pkl/"
    raw_root = "/data/3d/waymo_v1/validation/TOP/pcd/"
    model_type = args.model_type
    if args.knob_type == "Res":
        infer_root = "/data/3d/kitti_sampled/training/post_infer/" + model_type + "/resolution/" # inference result
        data_root = "/data/3d/kitti_sampled/training/pre_infer/resolution/" # processed data
        result_root = "/data/3d/kitti_sampled/training/visualize/resolution/" if args.result_dir is None else args.result_dir
        knob_list = sorted(os.listdir(infer_root), reverse=True)
    elif args.knob_type == "Edge":
        infer_root = "/data/3d/kitti_sampled/training/post_infer/" + model_type + "/edge_knob/"
        data_root = "/data/3d/kitti_sampled/training/pre_infer/edge_knob/"
        result_root = "/data/3d/kitti_sampled/training/visualize/edge_knob/" if args.result_dir is None else args.result_dir
        knob_list = sorted(os.listdir(infer_root), key=lambda x: float(x))
    elif args.knob_type == "Comb":
        infer_root = "/data/3d/kitti_sampled/training/post_infer/" + model_type + "/comb_knob/"
        data_root = "/data/3d/kitti_sampled/training/pre_infer/comb_knob/"
        result_root = "/data/3d/kitti_sampled/training/visualize/comb_knob/" if args.result_dir is None else args.result_dir
    elif args.knob_type == "FPS":
        infer_root = "/data/3d/kitti_sampled/training/baseline/post_infer/" + model_type + "/fps/"
        data_root = "/data/3d/kitti_sampled/training/baseline/pre_infer/fps/"
        result_root = "/data/3d/kitti_sampled/training/visualize/baseline/fps/" if args.result_dir is None else args.result_dir
        knob_list = sorted(os.listdir(infer_root), key=lambda x: float(x))
    elif args.knob_type == "RS":
        infer_root = "/data/3d/kitti_sampled/training/baseline/post_infer/" + model_type + "/rs/"
        data_root = "/data/3d/kitti_sampled/training/baseline/pre_infer/rs/"
        result_root = "/data/3d/kitti_sampled/training/visualize/baseline/rs/" if args.result_dir is None else args.result_dir
        knob_list = sorted(os.listdir(infer_root), key=lambda x: float(x))
        
    else: raise Exception("Invalid knob type")
    seq_list = range(50) if len(args.seq_list)==1 and args.seq_list[0] == "all" else [int(seq) for seq in args.seq_list]
    num_processors = min(multiprocessing.cpu_count(), args.num_threads)
    assert num_processors > 0
        
    os.makedirs(result_root, exist_ok=True)
    
    if args.knob_type == "Comb":
        res_knob_list = sorted(os.listdir(infer_root), key=lambda x: int(x))
        for res_knob_value in res_knob_list:
            res_infer_root = os.path.join(infer_root, res_knob_value)
            res_data_root = os.path.join(data_root, res_knob_value)
            res_result_root = os.path.join(result_root, res_knob_value)
            edge_knob_list = sorted(os.listdir(res_infer_root), key=lambda x: int(x))
            
            if num_processors == 1: # single thread
                for seq in seq_list:
                    seq_result(seq, edge_knob_list, raw_root, res_data_root, res_infer_root, gt_root, res_result_root)
            else: # multi-processing
                pool = multiprocessing.Pool(processes=num_processors)
                seq_args = [(seq, edge_knob_list, raw_root, res_data_root, res_infer_root, gt_root, res_result_root) for seq in seq_list]
                pool.starmap(seq_result, seq_args)
    else:
        if num_processors == 1: # single thread
            for seq in seq_list:
                seq_result(seq, knob_list, raw_root, data_root, infer_root, gt_root, result_root)
        else: # multi-processing
            pool = multiprocessing.Pool(processes=num_processors)
            seq_args = [(seq, knob_list, raw_root, data_root, infer_root, gt_root, result_root) for seq in seq_list]
            pool.starmap(seq_result, seq_args)


def seq_result(seq: int, knob_list: List[str], raw_root: str, data_root: str,
               infer_root: str, gt_root: str, result_root: str) -> None:
    numPoints = {knob_value: 0 for knob_value in knob_list}
    numPoints['raw'] = 0
    confidence_matrix = {}
    confidence_matrix_0 = {}
    confidence_matrix_20 = {}
    confidence_matrix_40 = {}
    mIous = {}
    
    space_saving = {}
    latency_avg = {}
    f1_scores = {}
    f1_scores_0 = {}
    f1_scores_20 = {}
    f1_scores_40 = {}

    # get raw point cloud size
    raw_path = os.path.join(raw_root, f"seq_{seq}")
    for frame in os.listdir(raw_path):
        numPoints['raw'] += int(linecache.getline(os.path.join(raw_path, frame), 7)[7:])
    assert numPoints["raw"] > 0

    for knob_value in knob_list:
        # if knob_value not in os.listdir(data_root):
        #     continue
        num_point_path = os.path.join(data_root, knob_value, f"seq_{seq}", "num_points.pkl")
        if os.path.isfile(num_point_path):
            with open(num_point_path, 'rb') as f:
                numPoints[knob_value] += sum(pickle.load(f))
            numPoints_acquired = True
        else:
            numPoints_acquired = False

        confidence_matrix[knob_value] = {'fn': 0, 'fp': 0, 'tp': 0}
        confidence_matrix_0[knob_value] = {'fn': 0, 'fp': 0, 'tp': 0}
        confidence_matrix_20[knob_value] = {'fn': 0, 'fp': 0, 'tp': 0}
        confidence_matrix_40[knob_value] = {'fn': 0, 'fp': 0, 'tp': 0}
        latency_avg[knob_value] = 0
        latency_knob = 0
        frame_cnt = 0
        
        mIous[knob_value] = []
        for frame in range(200):
            infer_path = os.path.join(infer_root, knob_value, f"seq_{seq}", f"{frame:0>5}.pkl")
            if args.knob_type == "Edge":
                data_path = os.path.join(data_root, f"diff_threshold_{knob_value}.0", f"seq_{seq}", f"{frame:0>5}.bin")
            else:
                data_path = os.path.join(data_root, knob_value, f"seq_{seq}", f"{frame:0>5}.bin")
            gt_path = os.path.join(gt_root, f"seq_{seq}", "annos", f"{frame:0>5}.pkl")
            result_path = os.path.join(result_root, f"seq_{seq}", knob_value, f"{frame:0>5}.png")
            if not os.path.isfile(infer_path) or not os.path.isfile(data_path) or not os.path.isfile(gt_path):
                continue

            with open(infer_path, 'rb') as f:
                infer, conf, label, latency = pickle.load(f)
            
            latency_knob += latency
            frame_cnt += 1
            gt = get_gt(gt_path)

            box_corners, confidences = infer_result_to_corners(infer, conf, label)
            fn, fp, tp, ious, mIou = get_confidence_matrix_orientation(box_corners, gt)
            
            confidence_matrix[knob_value]['fn'] += fn[0]
            confidence_matrix[knob_value]['fp'] += fp[0]
            confidence_matrix[knob_value]['tp'] += tp[0]
            confidence_matrix_0[knob_value]['fn'] += fn[1]
            confidence_matrix_0[knob_value]['fp'] += fp[1]
            confidence_matrix_0[knob_value]['tp'] += tp[1]
            confidence_matrix_20[knob_value]['fn'] += fn[2]
            confidence_matrix_20[knob_value]['fp'] += fp[2]
            confidence_matrix_20[knob_value]['tp'] += tp[2]
            confidence_matrix_40[knob_value]['fn'] += fn[3]
            confidence_matrix_40[knob_value]['fp'] += fp[3]
            confidence_matrix_40[knob_value]['tp'] += tp[3]
            
            mIous[knob_value].append(mIou)
            frame_result_path = os.path.join(result_root, f'seq_{seq}', knob_value, 'frame_result')
            os.makedirs(frame_result_path, exist_ok=True)
            
            # stack fn, fp, tp list to a numpy matrix
            np.save(os.path.join(frame_result_path, f"{frame}.npy"), np.stack((fn, fp, tp), axis=1))
            np.save(os.path.join(frame_result_path, f"{frame}_latency.npy"), latency)
            

            if not numPoints_acquired or args.save_plots:
                processed = np.fromfile(data_path, 
                                        dtype=np.float32).reshape(-1,5)
                if not numPoints_acquired:
                    numPoints[knob_value] += processed.shape[0]
                if args.save_plots:
                    plot_infer_result(processed, box_corners, confidences, seq, frame, result_path, ious)
        # calculate space saving and f1 score of each knob value
        space_saving[knob_value] = 1 - numPoints[knob_value] / numPoints['raw']
        f1_scores[knob_value] = get_F1(confidence_matrix[knob_value]['tp'], confidence_matrix[knob_value]['fp'], confidence_matrix[knob_value]['fn'])
        f1_scores_0[knob_value] = get_F1(confidence_matrix_0[knob_value]['tp'], confidence_matrix_0[knob_value]['fp'], confidence_matrix_0[knob_value]['fn'])
        f1_scores_20[knob_value] = get_F1(confidence_matrix_20[knob_value]['tp'], confidence_matrix_20[knob_value]['fp'], confidence_matrix_20[knob_value]['fn'])
        f1_scores_40[knob_value] = get_F1(confidence_matrix_40[knob_value]['tp'], confidence_matrix_40[knob_value]['fp'], confidence_matrix_40[knob_value]['fn'])
        if frame_cnt > 0:
            latency_avg[knob_value] = latency_knob / frame_cnt
        else:
            latency_avg[knob_value] = 0


    # plot space saving and f1 score of each sequence, controlling knob
    seq_result_path = os.path.join(result_root, f'seq_{seq}', 'result')
    os.makedirs(seq_result_path, exist_ok=True)

    plt.figure()
    plt.plot(list(space_saving.values()), list(f1_scores.values()), '-o')
    plt.xlabel('Space saving')
    plt.ylabel('F1 score')
    plt.title(f'Seq {seq}')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(os.path.join(seq_result_path, 'f1_score.png'))
    plt.close()
    
    plt.plot(list(space_saving.values()), list(f1_scores_0.values()), '-o', label='0-20m')
    plt.plot(list(space_saving.values()), list(f1_scores_20.values()), '-o', label='20-40m')
    plt.plot(list(space_saving.values()), list(f1_scores_40.values()), '-o', label='40m-')
    plt.title(f"Sequence {seq} for each range")
    plt.xlabel('Space saving')
    plt.ylabel('F1 score')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(os.path.join(seq_result_path, 'f1_score_range.png'))
    plt.close()
    
    # save results
    with open(os.path.join(seq_result_path, 'space_saving.pkl'), 'wb') as f:
        pickle.dump(space_saving, f)
    with open(os.path.join(seq_result_path, 'latency.pkl'), 'wb') as f:
        pickle.dump(latency_avg, f)
    with open(os.path.join(seq_result_path, 'f1_score.pkl'), 'wb') as f:
        pickle.dump(f1_scores, f)
    with open(os.path.join(seq_result_path, 'f1_score_0.pkl'), 'wb') as f:
        pickle.dump(f1_scores_0, f)
    with open(os.path.join(seq_result_path, 'f1_score_20.pkl'), 'wb') as f:
        pickle.dump(f1_scores_20, f)
    with open(os.path.join(seq_result_path, 'f1_score_40.pkl'), 'wb') as f:
        pickle.dump(f1_scores_40, f)
    with open(os.path.join(seq_result_path, 'mIous.pkl'), 'wb') as f:
        pickle.dump(mIous, f)
    with open(os.path.join(seq_result_path, 'conf_mat.pkl'), 'wb') as f:
        pickle.dump(confidence_matrix, f)
    with open(os.path.join(seq_result_path, 'conf_mat_0.pkl'), 'wb') as f:
        pickle.dump(confidence_matrix_0, f)
    with open(os.path.join(seq_result_path, 'conf_mat_20.pkl'), 'wb') as f:
        pickle.dump(confidence_matrix_20, f)
    with open(os.path.join(seq_result_path, 'conf_mat_40.pkl'), 'wb') as f:
        pickle.dump(confidence_matrix_40, f)
    with open(os.path.join(seq_result_path, 'numPoints.pkl'), 'wb') as f:
        pickle.dump(numPoints, f)

    print(f"Seq {seq} done")


def infer_result_to_corners(infer: np.ndarray, conf: np.ndarray, label: np.ndarray, cars_only: bool =True) \
                            -> Tuple[np.ndarray, np.ndarray]:
    box_corners = []
    confidences = []
    for box, confidence, lab in zip(infer, conf, label):
        if cars_only and lab!=0: continue
        if confidence < args.conf_thr: continue 
        box_corner = transform_to_corners(box[0], box[1], box[2], box[3], box[4], box[5], box[6], box_type="infer")
        box_corners.append(box_corner)
        confidences.append(confidence)
    box_corners = np.asarray(box_corners).reshape(-1, 8, 3)
    confidences = np.asarray(confidences)
    return box_corners, confidences


def get_confidence_matrix(box_corners: np.ndarray, gt: np.ndarray) -> Tuple[List[int], List[int], List[int], np.ndarray, float]:
    if gt.shape[0] == 0:
        fp_range_list = np.linalg.norm(np.mean(box_corners, axis=1), axis=1)
        fp_0, fp_20, fp_40 = range_checker(fp_range_list)
        return [0,0,0,0], [box_corners.shape[0], fp_0, fp_20, fp_40], [0,0,0,0], np.zeros((0, box_corners.shape[0])), 0
    if box_corners.shape[0] == 0:
        fn_range_list = np.linalg.norm(np.mean(gt, axis=1), axis=1)
        fn_0, fn_20, fn_40 = range_checker(fn_range_list)
        return [gt.shape[0], fn_0, fn_20, fn_40], [0,0,0,0], [0,0,0,0], np.zeros((gt.shape[0], 0)), 0   
    ious = box3d_overlap(torch.from_numpy(gt.astype(np.float32)), 
                         torch.from_numpy(box_corners.astype(np.float32)))[1].numpy()
    mIou = calc_mIoU(gt.astype(np.float32), box_corners.astype(np.float32), args.iou_thr)
    ious_cf = np.where(ious>args.iou_thr, ious, 0)
    
    # check each row of ious_cf, if more than two nonzero elements, make smaller one as zero value
    for i in range(ious_cf.shape[0]):
        nonzero = np.nonzero(ious_cf[i])[0]
        if len(nonzero) > 1:
            max_index = np.argmax(ious_cf[i])
            for j in nonzero:
                if j != max_index:
                    ious_cf[i][j] = 0
            
    zero_rows = np.all(ious_cf==0, axis=1)
    tp = np.nonzero(ious_cf)[0].shape[0]
    fn = ious_cf.shape[0] - tp
    fp = ious_cf.shape[1] - tp
    fn_range_list = np.linalg.norm(np.mean(gt[zero_rows], axis=1), axis=1)
    tp_range_list = np.linalg.norm(np.mean(gt[~zero_rows], axis=1), axis=1)
    fp_range_list = np.linalg.norm(np.mean(box_corners[np.all(ious_cf==0, axis=0)], axis=1), axis=1)

    assert fn == len(fn_range_list) and fp == len(fp_range_list) and tp == len(tp_range_list)
        
    fn_0, fn_20, fn_40 = range_checker(fn_range_list)
    tp_0, tp_20, tp_40 = range_checker(tp_range_list)
    fp_0, fp_20, fp_40 = range_checker(fp_range_list)
    
    FN = [fn, fn_0, fn_20, fn_40]
    FP = [fp, fp_0, fp_20, fp_40]
    TP = [tp, tp_0, tp_20, tp_40]
    
    return FN, FP, TP, ious, mIou

def get_confidence_matrix_orientation(box_corners: np.ndarray, gt: np.ndarray) -> Tuple[List[int], List[int], List[int], np.ndarray, float]:
    if gt.shape[0] == 0:
        box_center = np.mean(box_corners, axis=1)
        box_front = np.abs(box_center[:, 1]) <= np.abs(box_center[:, 0])
        fp_range_list_front = np.linalg.norm(np.mean(box_corners[box_front], axis=1), axis=1)
        fp_0, fp_20, fp_40 = range_checker(fp_range_list_front)
        return -1
        return [0,0,0,0], [box_corners[box_front].shape[0], fp_0, fp_20, fp_40], [0,0,0,0], np.zeros((0, box_corners.shape[0])), 0
    if box_corners.shape[0] == 0:
        fn_range_list = np.linalg.norm(np.mean(gt, axis=1), axis=1)
        fn_0, fn_20, fn_40 = range_checker(fn_range_list)
        return -1
        return [gt.shape[0], fn_0, fn_20, fn_40], [0,0,0,0], [0,0,0,0], np.zeros((gt.shape[0], 0)), 0   
    
    
    
    gt_center = np.mean(gt, axis=1)
    gt_front = np.abs(gt_center[:, 1]) <= np.abs(gt_center[:, 0])
    
    box_center = np.mean(box_corners, axis=1)
    box_front = np.abs(box_center[:, 1]) <= np.abs(box_center[:, 0])
    
    ious = box3d_overlap(torch.from_numpy(gt.astype(np.float32)), 
                         torch.from_numpy(box_corners.astype(np.float32)))[1].numpy()
    mIou = calc_mIoU(gt.astype(np.float32), box_corners.astype(np.float32), args.iou_thr)
    ious_cf = np.where(ious>args.iou_thr, ious, 0)
    
    # check each row of ious_cf, if more than two nonzero elements, make smaller one as zero value
    for i in range(ious_cf.shape[0]):
        nonzero = np.nonzero(ious_cf[i])[0]
        if len(nonzero) > 1:
            max_index = np.argmax(ious_cf[i])
            for j in nonzero:
                if j != max_index:
                    ious_cf[i][j] = 0
    
    ious_front = ious_cf[gt_front][:, box_front]
    ious_side = ious_cf[~gt_front][:, ~box_front]
    
    zero_rows_front = np.all(ious_front==0, axis=1)
    tp_front = np.nonzero(ious_front)[0].shape[0]
    fn_front = ious_front.shape[0] - tp_front
    fp_front = ious_front.shape[1] - tp_front
    fn_range_list_front = np.linalg.norm(np.mean(gt[gt_front][zero_rows_front], axis=1), axis=1)
    tp_range_list_front = np.linalg.norm(np.mean(gt[gt_front][~zero_rows_front], axis=1), axis=1)
    fp_range_list_front = np.linalg.norm(np.mean(box_corners[box_front][np.all(ious_front==0, axis=0)], axis=1), axis=1)
    
    fn_0_front, fn_20_front, fn_40_front = range_checker(fn_range_list_front)
    tp_0_front, tp_20_front, tp_40_front = range_checker(tp_range_list_front)
    fp_0_front, fp_20_front, fp_40_front = range_checker(fp_range_list_front)
    
    zero_rows_side = np.all(ious_side==0, axis=1)
    tp_side = np.nonzero(ious_side)[0].shape[0]
    fn_side = ious_side.shape[0] - tp_side
    fp_side = ious_side.shape[1] - tp_side
    fn_range_list_side = np.linalg.norm(np.mean(gt[~gt_front][zero_rows_side], axis=1), axis=1)
    tp_range_list_side = np.linalg.norm(np.mean(gt[~gt_front][~zero_rows_side], axis=1), axis=1)
    fp_range_list_side = np.linalg.norm(np.mean(box_corners[~box_front][np.all(ious_side==0, axis=0)], axis=1), axis=1)
    
    fn_0_side, fn_20_side, fn_40_side = range_checker(fn_range_list_side)
    tp_0_side, tp_20_side, tp_40_side = range_checker(tp_range_list_side)
    fp_0_side, fp_20_side, fp_40_side = range_checker(fp_range_list_side)
    
    FN_front = [fn_front, fn_0_front, fn_20_front, fn_40_front]
    FP_front = [fp_front, fp_0_front, fp_20_front, fp_40_front]
    TP_front = [tp_front, tp_0_front, tp_20_front, tp_40_front]
    FN_side = [fn_side, fn_0_side, fn_20_side, fn_40_side]
    FP_side = [fp_side, fp_0_side, fp_20_side, fp_40_side]
    TP_side = [tp_side, tp_0_side, tp_20_side, tp_40_side]
    
    return FN_front, FP_front, TP_front, FN_side, FP_side, TP_side, mIou
    


def plot_infer_result(processed: np.ndarray, box_corners: np.ndarray, confidences: np.ndarray,
                      seq: int, frame: int, result_path: str, ious: np.ndarray) -> None:
    raw_TP = np.load(f"/data/3d/kitti_sampled/training/post_infer/raw_data/seq_{seq}/TP_{frame}.npy")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    fig, ax = plt.subplots()
    point_cloud_range = [-74.88, -74.88, -4, 74.88, 74.88, 4]
    ax.scatter(processed[:, 0], processed[:, 1], s=0.01)
    plot_box(ax, raw_TP, 'raw_TP', color='g')
    plot_box(ax, box_corners, 'infer', alphas=confidences, ious=ious.max(axis=0))
    ax.legend(loc='upper right')
    plt.annotate(f"frame {frame}", (-74, -74))
    # plot origin point at (0,0)
    plt.xlim([point_cloud_range[1], point_cloud_range[4]])
    plt.ylim([point_cloud_range[0], point_cloud_range[3]])
    plt.savefig(result_path)
    plt.close()
    
    
def range_checker(range_list):
    v0, v1, v2 = 0, 0, 0
    for range in range_list:
        if 0 <= range and range <= 20:
            v0 += 1
        elif range <= 40:
            v1 += 1
        else:
            v2 += 1
    return v0, v1, v2    


def get_F1(TP, FP, FN):
    try:
        precision =  TP/(TP+FP)
        recall = TP/(TP+FN)
        return 2 / (1/precision + 1/recall)
    except ZeroDivisionError as e:
        print("Zero division!")
        print(f"    TP: {TP}, FP: {FP}, FN: {FN}")
        return 0
    

def get_gt(root: str, cars_only: bool =True, output_type: str ="corner") -> np.ndarray:
    assert output_type in ["center", "corner"]
    with open(root, 'rb') as fp:
        gt = pickle.load(fp)["objects"]
    gts_xyzwlha = []
    for gt_dict in gt:
        if cars_only and gt_dict["label"] != 1: continue
        gt_xyzwlha = np.zeros(7)
        gt_xyzwlha[:6] = gt_dict["box"][:6]
        gt_xyzwlha[-1] = gt_dict["box"][-1]
        if cars_only and gt_xyzwlha[3] > 8: continue
        if output_type == "corner":
            gt_xyzwlha = transform_to_corners(
                gt_xyzwlha[0], 
                gt_xyzwlha[1], 
                gt_xyzwlha[2], 
                gt_xyzwlha[4], 
                gt_xyzwlha[3], 
                gt_xyzwlha[5], 
                gt_xyzwlha[6], 
            )
        gts_xyzwlha.append(gt_xyzwlha)
    gts_xyzwlha = np.asarray(gts_xyzwlha).reshape(-1, 8, 3)
    return gts_xyzwlha


def plot_box(ax, box_corners, label, color='r', alphas=None, ious=None):
    if alphas is None:
        alphas = [1] * len(box_corners)
    for i, (box_3d, alpha) in enumerate(zip(box_corners, alphas)):
        start, end = 0, 1
        if ious is not None:
            if ious[i] < args.iou_thr:
                continue
            ax.text(box_3d[start, 0], box_3d[start, 1], f"{ious[i]:.2f}")
        ax.plot([box_3d[start,0], box_3d[end,0]], [box_3d[start,1], box_3d[end,1]], 
                color = color, label=label, alpha=alpha)
        label = None

        start, end = 1, 2
        ax.plot([box_3d[start,0], box_3d[end,0]], [box_3d[start,1], box_3d[end,1]], 
                color = color, label=label, alpha=alpha)

        start, end = 2, 3
        ax.plot([box_3d[start,0], box_3d[end,0]], [box_3d[start,1], box_3d[end,1]], 
                color = color, label=label, alpha=alpha)

        start, end = 3, 0
        ax.plot([box_3d[start,0], box_3d[end,0]], [box_3d[start,1], box_3d[end,1]], 
                color = color, label=label, alpha=alpha)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-t", "--knob-type", type=str, default="Res", choices=["Res", "Edge", "Comb", "FPS", "RS"])
    parser.add_argument("-r", "--result-dir", type=str)
    parser.add_argument("-c", "--conf-thr", type=float, default=0.5)
    parser.add_argument("-i", "--iou-thr", type=float, default=0.5)
    parser.add_argument("-n", "--num-threads", type=int, default=1)
    parser.add_argument("-p", "--save-plots", action="store_true")
    parser.add_argument("-m", "--model-type", type=str, default="parta2")
    parser.set_defaults(func=main)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    args.func(args)