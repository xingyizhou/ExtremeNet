import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import sys
import cv2
import numpy as np
import pickle
import json
ANN_PATH = '../data/coco/annotations/instances_val2017.json'
DEBUG = True

def _coco_box_to_bbox(box):
  bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                  dtype=np.int32)
  return bbox

def _overlap(box1, box2):
  area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
  inter = max(min(box1[2], box2[2]) - max(box1[0], box2[0]) + 1, 0) * \
        max(min(box1[3], box2[3]) - max(box1[1], box2[1]) + 1, 0)
  iou = 1.0 * inter / (area1 + 1e-5)
  return iou

def _box_inside(box2, box1):
    inside = (box2[0] >= box1[0] and box2[1] >= box1[1] and \
       box2[2] <= box1[2] and box2[3] <= box1[3])
    return inside

if __name__ == '__main__':
  if len(sys.argv) > 2:
    ANN_PATH = sys.argv[2]
  coco = coco.COCO(ANN_PATH)
  pred_path = sys.argv[1]
  out_path = pred_path[:-5] + '_no_ghost.json'
  dets = coco.loadRes(pred_path)
  img_ids = coco.getImgIds()
  num_images = len(img_ids)
  thresh = 4
  out = []
  for i, img_id in enumerate(img_ids):
    if i % 500 == 0:
      print(i)
    pred_ids = dets.getAnnIds(imgIds=[img_id])
    preds = dets.loadAnns(pred_ids)
    num_preds = len(preds)
    for j in range(num_preds):
      overlap_score = 0
      if preds[j]['score'] > 0.2:
        for k in range(num_preds):
          if  preds[j]['category_id'] == preds[k]['category_id'] and \
            _box_inside(_coco_box_to_bbox(preds[k]['bbox']), 
                        _coco_box_to_bbox(preds[j]['bbox'])) > 0.8:
            overlap_score += preds[k]['score']
        if overlap_score > thresh * preds[j]['score']:
          # print('overlap_score', overlap_score, preds[j]['score'])
          preds[j]['score'] = preds[j]['score'] / 2
          # preds[j]['score'] = preds[j]['score'] * np.exp(-(overlap_score / preds[j]['score'] - thresh)**2/2)
      out.append(preds[j])
  json.dump(out, open(out_path, 'w'))
  dets_refined = coco.loadRes(out_path)
  coco_eval = COCOeval(coco, dets_refined, "bbox")
  coco_eval.evaluate()
  coco_eval.accumulate()
  coco_eval.summarize()

  
