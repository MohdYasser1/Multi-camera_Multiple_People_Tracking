import argparse
from functools import partial
import os
import os.path as osp
import time
import cv2 
import json
import numpy as np
import sys
from util.camera import Camera
from Tracker.PoseTracker import Detection_Sample, PoseTracker, TrackState
from tqdm import tqdm
import copy

def main():
    print("start")
    scene_name = sys.argv[1]

    current_file_path = os.path.abspath(__file__)
    path_arr = current_file_path.split('/')[:-2]
    root_path = '/'.join(path_arr)

    det_dir = osp.join(root_path, 'result/detection', scene_name)
    pose_dir = osp.join(root_path, 'result/pose', scene_name)
    reid_dir = osp.join(root_path, 'result/reid', scene_name)

    cal_dir = osp.join(root_path, 'dataset/test')
    save_dir = osp.join(root_path, 'result/track')
    save_path = osp.join(save_dir, scene_name + '.txt')

    # cams = sorted(os.listdir(cal_dir))
    # cals = []
    # for cam in cams:
    #     cals.append(Camera(osp.join(cal_dir, cam, "calibration.json")))
    # Load the calibration data
    with open(osp.join(cal_dir, "calibration.json"), "r") as f:
        calibration_data = json.load(f)

    # Create Camera objects
    cals = [Camera(sensor) for sensor in calibration_data["sensors"]]

    # Sort if needed (now using idx_int for proper numeric sorting)
    cals.sort(key=lambda x: x.idx_int)

    det_data = []
    files = sorted(os.listdir(det_dir))
    files = [f for f in files if f[0] == 'C']
    for f in files:
        if f[0]=='C':
            print(osp.join(det_dir, f))
            det_data.append(np.loadtxt(osp.join(det_dir, f), delimiter=","))

    pose_data = []
    files = sorted(os.listdir(pose_dir))
    files = [f for f in files if f[0] == 'C']
    for f in files:
        pose_data.append(np.loadtxt(osp.join(pose_dir, f)))

    reid_data = []
    files = sorted([f for f in os.listdir(reid_dir) if f.startswith('C')])
    for f in files:
        reid_file = osp.join(reid_dir, f)
        reid_data_scene = np.load(reid_file, mmap_mode='r')
        reid_data.append(reid_data_scene)

    print("reading finish")

    max_frame = []
    for det_sv in det_data:
        if len(det_sv):
            max_frame.append(np.max(det_sv[:, 0]))
    max_frame = int(np.max(max_frame))

    tracker = PoseTracker(cals)
    box_thred = 0.3
    results = []

    for frame_id in tqdm(range(max_frame+1), desc= scene_name):
        detection_sample_mv = []
        for v in range(tracker.num_cam):
            detection_sample_sv = []
            det_sv = det_data[v]
            if len(det_sv)==0:
                detection_sample_mv.append(detection_sample_sv)
                continue
            idx = det_sv[:, 0] == frame_id
            cur_det = det_sv[idx]
            curr_pose = pose_data[v][idx]
            curr_reid = reid_data[v][idx]

            for det, pose, reid in zip(cur_det, curr_pose, curr_reid):
                if det[-1]<box_thred or len(det)==0:
                    continue
                reid_normalized = reid / np.linalg.norm(reid)
                new_sample = Detection_Sample(bbox=det[2:], keypoints_2d=pose[6:].reshape(17,3), reid_feat=reid_normalized, cam_id=v, frame_id=frame_id)
                detection_sample_sv.append(new_sample)
            detection_sample_mv.append(detection_sample_sv)

        print("frame {}".format(frame_id), "det nums: ", [len(L) for L in detection_sample_mv])

        if frame_id == 231:
            print("frame 231")
        

        tracker.mv_update_wo_pred(detection_sample_mv, frame_id)

        frame_results = tracker.output(frame_id)
        results += frame_results

    results = np.concatenate(results, axis=0)
    sort_idx = np.lexsort((results[:, 0], results[:, 1]))
    results = np.ascontiguousarray(results[sort_idx])
    np.savetxt(save_path, results)

 

if __name__ == '__main__':
    main()

