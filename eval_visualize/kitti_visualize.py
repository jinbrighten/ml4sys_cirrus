import copy
import os

import cv2
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np

import config.kitti_config as cnf
from data_process import transformation
from data_process.kitti_bev_utils import drawRotatedBox, makeBEVMap
from data_process.kitti_data_utils import Calibration, get_filtered_lidar
from utils.visualization_utils import (merge_rgb_to_bev,
                                       show_rgb_image_with_boxes)

output_width = 608

#font = font_manager.FontProperties(family='Arial', weight='bold', style='normal', size=24)
font = font_manager.FontProperties(weight='bold', style='normal', size=12)
legend_font = font_manager.FontProperties(style='normal', size='xx-large')
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 24
#plt.rcParams['ytick.labelweight'] = 'bold'
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.titleweight'] = 'bold'

colors = ['r', 'b', 'g', 'k', 'y', 'm', 'c']
linestyles = ["-", ":", "-."]
markers = ["x", "o", "v", "^", "s", "s"]


def get_label_car(data_root, sample_idx):
    calib_path = os.path.join(data_root, "calib", "%06d.txt"%sample_idx) 
    calib = Calibration(calib_path)
    label_path = os.path.join(data_root, "label_2", "%06d.txt"%sample_idx) 
    
    labels = []
    for line in open(label_path, 'r'):
        # print(line)
        line = line.rstrip()
        line_parts = line.split(' ')

        obj_name = line_parts[0]  # 'Car', 'Pedestrian', ...
        cat_id = int(cnf.CLASS_NAME_TO_ID[obj_name])
        if cat_id != 1:  # ignore Tram and Misc
            continue
        truncated = int(float(line_parts[1]))  # truncated pixel ratio [0..1]
        occluded = int(line_parts[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        alpha = float(line_parts[3])  # object observation angle [-pi..pi]
        # xmin, ymin, xmax, ymax
        xmin, ymin, xmax, ymax = float(line_parts[4]), float(line_parts[5]), float(line_parts[6]), float(line_parts[7])
        # height, width, length (h, w, l)
        h, w, l = float(line_parts[8]), float(line_parts[9]), float(line_parts[10])
        # location (x,y,z) in camera coord.
        x, y, z = float(line_parts[11]), float(line_parts[12]), float(line_parts[13])
        ry = float(line_parts[14])  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

        object_label = [x, y, z, h, w, l, ry, xmin, ymin, xmax, ymax]
        labels.append(object_label)

    if len(labels) == 0:
        labels = np.zeros((1, 11), dtype=np.float32)
        has_labels = False
    else:
        labels = np.array(labels, dtype=np.float32)
        has_labels = True
    
    labels_lidar = copy.deepcopy(labels)
    if has_labels:
        labels_lidar[:, 0:7] = transformation.camera_to_lidar_box(labels[:, 0:7], calib.V2C, calib.R0, calib.P2)
    
    return labels_lidar, has_labels
    

def get_label(label_path):
    labels = []
    for line in open(label_path, 'r'):
        line = line.rstrip()
        line_parts = line.split(' ')

        obj_name = line_parts[0]  # 'Car', 'Pedestrian', ...
        cat_id = int(cnf.CLASS_NAME_TO_ID[obj_name])
        if cat_id <= -99:  # ignore Tram and Misc
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

        object_label = [cat_id, x, y, z, h, w, l, ry]
        labels.append(object_label)

    if len(labels) == 0:
        labels = np.zeros((1, 8), dtype=np.float32)
        has_labels = False
    else:
        labels = np.array(labels, dtype=np.float32)
        has_labels = True

        return labels, has_labels

    
def load_sample(data_root, sample_idx):
    lidar_path = os.path.join(data_root, "velodyne", "%06d.bin"%sample_idx) 
    lidar_data = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

    img_path = os.path.join(data_root, "image_2", "%06d.png"%sample_idx) 
    img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    calib_path = os.path.join(data_root, "calib", "%06d.txt"%sample_idx) 
    calib = Calibration(calib_path)

    label_path = os.path.join(data_root, "label_2", "%06d.txt"%sample_idx) 
    labels_rgb, has_labels  = get_label(label_path)
    
    labels_lidar = copy.deepcopy(labels_rgb)
    if has_labels:
        labels_lidar[:, 1:] = transformation.camera_to_lidar_box(labels_rgb[:, 1:], calib.V2C, calib.R0, calib.P2)

    lidar_data_filtered, _ = get_filtered_lidar(lidar_data, cnf.boundary, labels_lidar)
    
    return lidar_data, lidar_data_filtered, img_rgb, calib, labels_rgb, labels_lidar


def visualize_data(lidar_data, lidar_data_filtered, img_rgb, calib, labels_rgb, labels_lidar):
    bev_map = makeBEVMap(lidar_data_filtered, cnf.boundary, draw_half = cnf.draw_half)
    bev_map = (bev_map.transpose(1, 2, 0) * 255).astype(np.uint8)
    bev_map = cv2.resize(bev_map, (cnf.BEV_HEIGHT, cnf.BEV_WIDTH))
    
    figsize = (10,10)
    plt.figure(figsize = figsize)
    plt.imshow(cv2.rotate(bev_map, cv2.ROTATE_180))
    plt.axis("off")
    plt.show()

    for box_idx, (cls_id, x, y, z, h, w, l, yaw) in enumerate(labels_lidar):
        # Draw rotated box
        yaw = -yaw
        y1 = int((x - cnf.boundary['minX']) / cnf.DISCRETIZATION)
        x1 = int((y - cnf.boundary['minY']) / cnf.DISCRETIZATION)
        w1 = int(w / cnf.DISCRETIZATION)
        l1 = int(l / cnf.DISCRETIZATION)

        drawRotatedBox(bev_map, x1, y1, w1, l1, yaw, cnf.colors[int(cls_id)])
    # Rotate the bev_map
    bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

    img_rgb = show_rgb_image_with_boxes(img_rgb, labels_rgb, calib)
    plt.figure(figsize = figsize)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()
    
    out_img = merge_rgb_to_bev(img_rgb, bev_map, output_width=output_width)
    plt.figure(figsize = (10,20))
    plt.imshow(out_img)
    plt.axis("off")
    plt.show()
    print("number3")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(-lidar_data_filtered[:, 1], lidar_data_filtered[:, 0], color='k', s=1)
    
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print("number4")
    

def visualize_inference(inference_result, lidar_data, lidar_data_filtered, img_rgb, calib, labels_rgb, labels_lidar):
    
    bev_map = makeBEVMap(lidar_data_filtered, cnf.boundary, draw_half = cnf.draw_half)
    bev_map = (bev_map.transpose(1, 2, 0) * 255).astype(np.uint8)
    bev_map = cv2.resize(bev_map, (cnf.BEV_HEIGHT, cnf.BEV_WIDTH))
    figsize = (10,10)
    
    print("GT bbox")
    for box_idx, (cls_id, x, y, z, h, w, l, yaw) in enumerate(labels_lidar):
        # Draw rotated box
        yaw = -yaw
        y1 = int((x - cnf.boundary['minX']) / cnf.DISCRETIZATION)
        x1 = int((y - cnf.boundary['minY']) / cnf.DISCRETIZATION)
        w1 = int(w / cnf.DISCRETIZATION)
        l1 = int(l / cnf.DISCRETIZATION)

        print("draw %d bbox, x: %d y: %d w: %d l: %d yaw: %4f cls: %d"%(box_idx, x1, y1, w1, l1, yaw, cls_id))
        drawRotatedBox(bev_map, x1, y1, w1, l1, yaw, cnf.colors[int(cls_id)])
        
    print("\n\nPredicted bbox")
    for box_idx, (x, y, z, h, w, l, yaw) in enumerate(inference_result):
        # Draw rotated box
        yaw = -yaw
        y1 = int((x - cnf.boundary['minX']) / cnf.DISCRETIZATION)
        x1 = int((y - cnf.boundary['minY']) / cnf.DISCRETIZATION)
        w1 = int(w / cnf.DISCRETIZATION)
        l1 = int(l / cnf.DISCRETIZATION)

        print("draw %d bbox, x: %d y: %d w: %d l: %d yaw: %4f"%(box_idx, x1, y1, w1, l1, yaw))
        drawRotatedBox(bev_map, x1, y1, w1, l1, yaw, (255, 255, 255))
    # Rotate the bev_map
    bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

    img_rgb = show_rgb_image_with_boxes(img_rgb, labels_rgb, calib)
    out_img = merge_rgb_to_bev(img_rgb, bev_map, output_width=output_width)
    plt.figure(figsize = (10,20))
    plt.imshow(out_img)
    legend_labels = ['predicted', 'GT']
    legend_markers = ['o', 'o']
    legend_colors = ['white', 'blue']
    plt.legend(loc='lower right', handles=[plt.Line2D([], [], color=legend_colors[i], marker=legend_markers[i], linestyle='None', label=legend_labels[i]) for i in range(2)])
    plt.axis("off")
    plt.show()
    

def process_sample(data_root, sample_idx, verbose = True):
    '''
    The mapping between name and cls_ID is as follows.
        Pedestrian      : 0
        Car             : 1
        Cyclist         : 2
        Van             : 1
        Truck           : -3
        Person_sitting  : 0
        Tram            : -99
        Misc            : -99
        DontCare        : -1
    '''
    lidar_data, lidar_data_filtered, img_rgb, calib, labels_rgb, labels_lidar = load_sample(data_root, sample_idx)
    return lidar_data, lidar_data_filtered, img_rgb, calib, labels_rgb, labels_lidar


def visualize_sample(data_root, sample_idx, lidar_data, inference_result = -1, inference = False):
    _, _, img_rgb, calib, labels_rgb, labels_lidar = load_sample(data_root, sample_idx)
    lidar_data_filtered = get_filtered_lidar(lidar_data, cnf.boundary)

    if inference:
        visualize_inference(inference_result, lidar_data, lidar_data_filtered, img_rgb, calib, labels_rgb, labels_lidar)
    else:
        visualize_data(lidar_data, lidar_data_filtered, img_rgb, calib, labels_rgb, labels_lidar)
        
        
        
        
def visualize_inference_bev(inference_result, lidar_data_filtered, labels_lidar):
    
    bev_map = makeBEVMap(lidar_data_filtered, cnf.boundary, draw_half = cnf.draw_half)
    bev_map = (bev_map.transpose(1, 2, 0) * 255).astype(np.uint8)
    bev_map = cv2.resize(bev_map, (cnf.BEV_HEIGHT, cnf.BEV_WIDTH))
    figsize = (10,10)
    
    print("GT bbox")
    for box_idx, (x, y, z, l, w, h, yaw) in enumerate(labels_lidar):
        # Draw rotated box
        yaw = -yaw
        y1 = int((x - cnf.boundary['minX']) / cnf.DISCRETIZATION)
        x1 = int((y - cnf.boundary['minY']) / cnf.DISCRETIZATION)
        w1 = int(w / cnf.DISCRETIZATION)
        l1 = int(l / cnf.DISCRETIZATION)

        print("draw %d bbox, x: %d y: %d z: %d h: %d w: %d l: %d yaw: %4f"%(box_idx, x1, y1, z, h, w1, l1, yaw))
        if box_idx == 0:
            drawRotatedBox(bev_map, x1, y1, w1, l1, yaw, cnf.colors[int(1)])
        
    print("\n\nPredicted bbox")
    for box_idx, (x, y, z, l, w, h, yaw) in enumerate(inference_result):
        # Draw rotated box
        yaw = -yaw
        y1 = int((x - cnf.boundary['minX']) / cnf.DISCRETIZATION)
        x1 = int((y - cnf.boundary['minY']) / cnf.DISCRETIZATION)
        w1 = int(w / cnf.DISCRETIZATION)
        l1 = int(l / cnf.DISCRETIZATION)

        print("draw %d bbox, x: %d y: %d z: %d h: %d w: %d l: %d yaw: %4f"%(box_idx, x1, y1, z, h, w1, l1, yaw))
        drawRotatedBox(bev_map, x1, y1, w1, l1, yaw, (255, 255, 255))
    # Rotate the bev_map
    bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

    plt.figure(figsize = (10,20))
    plt.imshow(bev_map)
    legend_labels = ['predicted', 'GT']
    legend_markers = ['o', 'o']
    legend_colors = ['white', 'blue']
    plt.legend(loc='lower right', handles=[plt.Line2D([], [], color=legend_colors[i], marker=legend_markers[i], linestyle='None', label=legend_labels[i]) for i in range(2)])
    plt.axis("off")
    plt.show()
    