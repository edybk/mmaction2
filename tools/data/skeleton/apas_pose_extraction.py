# Copyright (c) OpenMMLab. All rights reserved.
import abc
import argparse
import os
import os.path as osp
import random as rd
import shutil
import string
import warnings
from collections import defaultdict

import cv2
import mmcv
import numpy as np

try:
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except ImportError:
    warnings.warn(
        'Please install MMDet and MMPose for NTURGB+D pose extraction.'
    )  # noqa: E501



mmdet_root = '/data/home/bedward/workspace/mmpose-project/mmdetection'
mmpose_root = '/data/home/bedward/workspace/mmpose-project/mmpose'

args = abc.abstractproperty()
args.det_config = f'{mmdet_root}/configs/yolox/yolox_s_8x8_300e_apas.py'  
args.det_checkpoint = f'{mmdet_root}/work_dirs/yolox_s_8x8_300e_apas/epoch_650.pth'
args.det_score_thr = 0.5
args.pose_config = f'{mmpose_root}/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/apas/res50_apas_224x224.py'
args.pose_checkpoint = f'{mmpose_root}/work_dirs/res50_apas_224x224/best_AUC_epoch_59.pth'

labels_dict = {
  "G0": 0,
  "G1": 1,
  "G2": 2,
  "G3": 3,
  "G4": 4,
  "G5": 5,
  "G6": 6,
  "G7": 7
}

def gen_id(size=8):
    chars = string.ascii_uppercase + string.digits
    return ''.join(rd.choice(chars) for _ in range(size))


tmp_dname = f"tmp_{gen_id()}"
print(f"tmp folder name = {tmp_dname}")

def extract_frame(video_path):
    dname = gen_id()
    frame_dir = f"{tmp_dname}/{dname}"
    os.makedirs(frame_dir, exist_ok=True)
    frame_tmpl = osp.join(frame_dir, 'img_{:05d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    while flag:
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths

detector_model = None
def get_detector_model(config, checkpoint, device):
    global detector_model
    if not detector_model:
        detector_model = init_detector(config, checkpoint, device)
        print("initialized detector model")
    return detector_model

def detection_inference(args, frame_paths):
    model = get_detector_model(args.det_config, args.det_checkpoint, args.device)
    # assert model.CLASSES[0] == 'person', ('We require you to use a detector '
    #                                       'trained on COCO')
    results = []
    print('Performing Hand Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def intersection(b0, b1):
    l, r = max(b0[0], b1[0]), min(b0[2], b1[2])
    u, d = max(b0[1], b1[1]), min(b0[3], b1[3])
    return max(0, r - l) * max(0, d - u)


def iou(b0, b1):
    i = intersection(b0, b1)
    u = area(b0) + area(b1) - i
    return i / u


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def removedup(bbox):

    def inside(box0, box1, thre=0.8):
        return intersection(box0, box1) / area(box0) > thre

    num_bboxes = bbox.shape[0]
    if num_bboxes == 1 or num_bboxes == 0:
        return bbox
    valid = []
    for i in range(num_bboxes):
        flag = True
        for j in range(num_bboxes):
            if i != j and inside(bbox[i],
                                 bbox[j]) and bbox[i][4] <= bbox[j][4]:
                flag = False
                break
        if flag:
            valid.append(i)
    return bbox[valid]


def is_easy_example(det_results, num_person):
    threshold = 0.95

    def thre_bbox(bboxes, thre=threshold):
        shape = [sum(bbox[:, -1] > thre) for bbox in bboxes]
        ret = np.all(np.array(shape) == shape[0])
        return shape[0] if ret else -1

    if thre_bbox(det_results) == num_person:
        det_results = [x[x[..., -1] > 0.95] for x in det_results]
        return True, np.stack(det_results)
    return False, thre_bbox(det_results)


def bbox2tracklet(bbox):
    iou_thre = 0.6
    tracklet_id = -1
    tracklet_st_frame = {}
    tracklets = defaultdict(list)
    for t, box in enumerate(bbox):
        for idx in range(box.shape[0]):
            matched = False
            for tlet_id in range(tracklet_id, -1, -1):
                cond1 = iou(tracklets[tlet_id][-1][-1], box[idx]) >= iou_thre
                cond2 = (
                    t - tracklet_st_frame[tlet_id] - len(tracklets[tlet_id]) <
                    10)
                cond3 = tracklets[tlet_id][-1][0] != t
                if cond1 and cond2 and cond3:
                    matched = True
                    tracklets[tlet_id].append((t, box[idx]))
                    break
            if not matched:
                tracklet_id += 1
                tracklet_st_frame[tracklet_id] = t
                tracklets[tracklet_id].append((t, box[idx]))
    return tracklets


def drop_tracklet(tracklet):
    tracklet = {k: v for k, v in tracklet.items() if len(v) > 5}

    def meanarea(track):
        boxes = np.stack([x[1] for x in track]).astype(np.float32)
        areas = (boxes[..., 2] - boxes[..., 0]) * (
            boxes[..., 3] - boxes[..., 1])
        return np.mean(areas)

    tracklet = {k: v for k, v in tracklet.items() if meanarea(v) > 5000}
    return tracklet


def distance_tracklet(tracklet):
    dists = {}
    for k, v in tracklet.items():
        bboxes = np.stack([x[1] for x in v])
        c_x = (bboxes[..., 2] + bboxes[..., 0]) / 2.
        c_y = (bboxes[..., 3] + bboxes[..., 1]) / 2.
        c_x -= 480
        c_y -= 270
        c = np.concatenate([c_x[..., None], c_y[..., None]], axis=1)
        dist = np.linalg.norm(c, axis=1)
        dists[k] = np.mean(dist)
    return dists


def tracklet2bbox(track, num_frame):
    # assign_prev
    bbox = np.zeros((num_frame, 5))
    trackd = {}
    for k, v in track:
        bbox[k] = v
        trackd[k] = v
    for i in range(num_frame):
        if bbox[i][-1] <= 0.5:
            mind = np.Inf
            for k in trackd:
                if np.abs(k - i) < mind:
                    mind = np.abs(k - i)
            bbox[i] = bbox[k]
    return bbox


def tracklets2bbox(tracklet, num_frame):
    dists = distance_tracklet(tracklet)
    sorted_inds = sorted(dists, key=lambda x: dists[x])
    dist_thre = np.Inf
    for i in sorted_inds:
        if len(tracklet[i]) >= num_frame / 2:
            dist_thre = 2 * dists[i]
            break

    dist_thre = max(50, dist_thre)

    bbox = np.zeros((num_frame, 5))
    bboxd = {}
    for idx in sorted_inds:
        if dists[idx] < dist_thre:
            for k, v in tracklet[idx]:
                if bbox[k][-1] < 0.01:
                    bbox[k] = v
                    bboxd[k] = v
    bad = 0
    for idx in range(num_frame):
        if bbox[idx][-1] < 0.01:
            bad += 1
            mind = np.Inf
            mink = None
            for k in bboxd:
                if np.abs(k - idx) < mind:
                    mind = np.abs(k - idx)
                    mink = k
            bbox[idx] = bboxd[mink]
    return bad, bbox


def bboxes2bbox(bbox, num_frame):
    ret = np.zeros((num_frame, 2, 5))
    for t, item in enumerate(bbox):
        if item.shape[0] <= 2:
            ret[t, :item.shape[0]] = item
        else:
            inds = sorted(
                list(range(item.shape[0])), key=lambda x: -item[x, -1])
            ret[t] = item[inds[:2]]
    for t in range(num_frame):
        if ret[t, 0, -1] <= 0.01:
            ret[t] = ret[t - 1]
        elif ret[t, 1, -1] <= 0.01:
            if t:
                if ret[t - 1, 0, -1] > 0.01 and ret[t - 1, 1, -1] > 0.01:
                    if iou(ret[t, 0], ret[t - 1, 0]) > iou(
                            ret[t, 0], ret[t - 1, 1]):
                        ret[t, 1] = ret[t - 1, 1]
                    else:
                        ret[t, 1] = ret[t - 1, 0]
    return ret


def ntu_det_postproc(vid, det_results):
    det_results = [removedup(x) for x in det_results]
    label = int(vid.split('/')[-1].split('A')[1][:3])
    mpaction = list(range(50, 61)) + list(range(106, 121))
    n_person = 2 if label in mpaction else 1
    is_easy, bboxes = is_easy_example(det_results, n_person)
    if is_easy:
        print('\nEasy Example')
        return bboxes

    tracklets = bbox2tracklet(det_results)
    tracklets = drop_tracklet(tracklets)

    print(f'\nHard {n_person}-person Example, found {len(tracklets)} tracklet')
    if n_person == 1:
        if len(tracklets) == 1:
            tracklet = list(tracklets.values())[0]
            det_results = tracklet2bbox(tracklet, len(det_results))
            return np.stack(det_results)
        else:
            bad, det_results = tracklets2bbox(tracklets, len(det_results))
            return det_results
    # n_person is 2
    if len(tracklets) <= 2:
        tracklets = list(tracklets.values())
        bboxes = []
        for tracklet in tracklets:
            bboxes.append(tracklet2bbox(tracklet, len(det_results))[:, None])
        bbox = np.concatenate(bboxes, axis=1)
        return bbox
    else:
        return bboxes2bbox(det_results, len(det_results))

pose_model = None
def get_pose_model(config, checkpoint, device):
    global pose_model
    if not pose_model:
        pose_model = init_pose_model(config, checkpoint, device)
        print("initialized pose model")
    return pose_model

def pose_inference(args, frame_paths, det_results):
    model = get_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    print('Performing Hand Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))

    num_frame = len(det_results)
    num_hands = max([len(x) for x in det_results])
    kp = np.zeros((num_hands, num_frame, 21, 3), dtype=np.float32)

    for i, (f, d) in enumerate(zip(frame_paths, det_results)):
        # Align input format
        d = [dict(bbox=x) for x in list(d) if x[-1] > 0.5]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        for j, item in enumerate(pose):
            kp[j, i] = item['keypoints']
        prog_bar.update()
    return kp


def apas_pose_extraction(vid, skip_postproc=False):
    frame_paths = extract_frame(vid)
    det_results = detection_inference(args, frame_paths)
    if not skip_postproc:
        det_results = ntu_det_postproc(vid, det_results)
    pose_results = pose_inference(args, frame_paths, det_results)
    anno = dict()
    anno['keypoint'] = pose_results[..., :2]
    anno['keypoint_score'] = pose_results[..., 2]
    anno['frame_dir'] = osp.splitext(osp.basename(vid))[0]
    anno['img_shape'] = (1080, 1920)
    anno['original_shape'] = (1080, 1920)
    anno['total_frames'] = pose_results.shape[1]
    anno['label'] = labels_dict[os.path.dirname(vid).split('/')[-1]]
    shutil.rmtree(osp.dirname(frame_paths[0]))

    return anno


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Pose Annotation for a single NTURGB-D video')
    parser.add_argument('video', type=str, help='source video')
    parser.add_argument('output', type=str, help='output pickle name')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--skip-postproc', action='store_true')
    parser.add_argument('--whole-dir', action='store_true')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    global_args = parse_args()
    args.device = global_args.device
    args.video = global_args.video
    args.output = global_args.output
    args.skip_postproc = global_args.skip_postproc
    args.whole_dir = global_args.whole_dir
    if not args.whole_dir:
        anno = apas_pose_extraction(args.video, args.skip_postproc)
        mmcv.dump(anno, args.output)
    else:
        vid_root = args.video
        label_folders = [o for o in os.listdir(vid_root) 
                    if os.path.isdir(os.path.join(vid_root,o))]
        for label in label_folders:
            print(f"processing vids of {label}")
            os.makedirs(f"{args.output}/{label}", exist_ok=True)
            for vid in os.listdir(f"{vid_root}/{label}"):
                vid_path = f"{vid_root}/{label}/{vid}"
                print(f"processing {vid_path}")
                out_path = f"{args.output}/{label}/{vid[:-4]}.pkl"
                try:
                    anno = apas_pose_extraction(vid_path, args.skip_postproc)
                    mmcv.dump(anno, out_path)
                except:
                    print(f"failure at vid {vid_path}, skipping")
    shutil.rmtree(tmp_dname)