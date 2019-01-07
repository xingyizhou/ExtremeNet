import pycocotools.coco as cocoapi
import sys
import cv2
import numpy as np
import pickle
import json
SPLITS = ['val', 'train']
ANN_PATH = '../data/coco/annotations/instances_{}2017.json'
OUT_PATH = '../data/coco/annotations/instances_extreme_{}2017.json'
IMG_DIR = '../data/coco/{}2017/'
DEBUG = False
from scipy.spatial import ConvexHull

def _coco_box_to_bbox(box):
  bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                  dtype=np.int32)
  return bbox

def _get_extreme_points(pts):
  l, t = min(pts[:, 0]), min(pts[:, 1])
  r, b = max(pts[:, 0]), max(pts[:, 1])
  # 3 degrees
  thresh = 0.02
  w = r - l + 1
  h = b - t + 1
  
  pts = np.concatenate([pts[-1:], pts, pts[:1]], axis=0)
  t_idx = np.argmin(pts[:, 1])
  t_idxs = [t_idx]
  tmp = t_idx + 1
  while tmp < pts.shape[0] and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
    t_idxs.append(tmp)
    tmp += 1
  tmp = t_idx - 1
  while tmp >= 0 and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
    t_idxs.append(tmp)
    tmp -= 1
  tt = [(max(pts[t_idxs, 0]) + min(pts[t_idxs, 0])) // 2, t]

  b_idx = np.argmax(pts[:, 1])
  b_idxs = [b_idx]
  tmp = b_idx + 1
  while tmp < pts.shape[0] and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
    b_idxs.append(tmp)
    tmp += 1
  tmp = b_idx - 1
  while tmp >= 0 and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
    b_idxs.append(tmp)
    tmp -= 1
  bb = [(max(pts[b_idxs, 0]) + min(pts[b_idxs, 0])) // 2, b]

  l_idx = np.argmin(pts[:, 0])
  l_idxs = [l_idx]
  tmp = l_idx + 1
  while tmp < pts.shape[0] and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
    l_idxs.append(tmp)
    tmp += 1
  tmp = l_idx - 1
  while tmp >= 0 and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
    l_idxs.append(tmp)
    tmp -= 1
  ll = [l, (max(pts[l_idxs, 1]) + min(pts[l_idxs, 1])) // 2]

  r_idx = np.argmax(pts[:, 0])
  r_idxs = [r_idx]
  tmp = r_idx + 1
  while tmp < pts.shape[0] and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
    r_idxs.append(tmp)
    tmp += 1
  tmp = r_idx - 1
  while tmp >= 0 and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
    r_idxs.append(tmp)
    tmp -= 1
  rr = [r, (max(pts[r_idxs, 1]) + min(pts[r_idxs, 1])) // 2]

  return np.array([tt, ll, bb, rr])

if __name__ == '__main__':
  for split in SPLITS:
    data = json.load(open(ANN_PATH.format(split), 'r'))
    coco = cocoapi.COCO(ANN_PATH.format(split))
    img_ids = coco.getImgIds()
    num_images = len(img_ids)
    num_classes = 80
    tot_box = 0
    print('num_images', num_images)
    anns_all = data['annotations']
    for i, ann in enumerate(anns_all):
      tot_box += 1
      bbox = ann['bbox']
      seg = ann['segmentation']
      if type(seg) == list:
        if len(seg) == 1:
          pts = np.array(seg[0]).reshape(-1, 2)
        else:
          pts = []
          for v in seg:
            pts += v
          pts = np.array(pts).reshape(-1, 2)
      else:
        mask = coco.annToMask(ann) * 255
        tmp = np.where(mask > 0)
        pts = np.asarray(tmp).transpose()[:, ::-1].astype(np.int32)
      extreme_points = _get_extreme_points(pts).astype(np.int32)
      anns_all[i]['extreme_points'] = extreme_points.copy().tolist()
      if DEBUG:
        img_id = ann['image_id']
        img_info = coco.loadImgs(ids=[img_id])[0]
        img_path = IMG_DIR.format(split) + img_info['file_name']
        img = cv2.imread(img_path)
        if type(seg) == list:
          mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
          cv2.fillPoly(mask, [pts.astype(np.int32).reshape(-1, 1, 2)], (255,0,0))
        else:
          mask = mask.reshape(img.shape[0], img.shape[1], 1)
        img = (0.4 * img + 0.6 * mask).astype(np.uint8)
        bbox = _coco_box_to_bbox(ann['bbox'])
        cl = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
        for j in range(extreme_points.shape[0]):
          cv2.circle(img, (extreme_points[j, 0], extreme_points[j, 1]),
                          5, cl[j], -1)
        cv2.imshow('img', img)
        cv2.waitKey()
    print('tot_box', tot_box)   
    data['annotations'] = anns_all
    json.dump(data, open(OUT_PATH.format(split), 'w'))
  

