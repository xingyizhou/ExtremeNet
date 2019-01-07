from dextr.dextr import Dextr
import pycocotools.coco as cocoapi
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask
import numpy as np
import sys
import cv2
import json
from progress.bar import Bar
DEBUG = False
ANN_PATH = 'data/coco/annotations/instances_extreme_val2017.json'
IMG_DIR = 'data/coco/images/val2017/'

if __name__ == '__main__':
    dextr = Dextr()
    coco = cocoapi.COCO(ANN_PATH)
    pred_path = sys.argv[1]
    out_path = pred_path[:-5] + '_segm.json'
    data = json.load(open(pred_path, 'r'))
    anns = data
    results = []
    score_thresh = 0.2
    num_boxes = 0
    for i, ann in enumerate(anns):
        if ann['score'] >= score_thresh:
            num_boxes += 1
    
    bar = Bar('Pred + Dextr', max=num_boxes)
    for i, ann in enumerate(anns):
        if ann['score'] < score_thresh:
            continue
        ex = np.array(ann['extreme_points'], dtype=np.int32).reshape(4, 2)
        img_id = ann['image_id']
        img_info = coco.loadImgs(ids=[img_id])[0]
        img_path = IMG_DIR + img_info['file_name']
        img = cv2.imread(img_path)
        mask = dextr.segment(img[:, :, ::-1], ex)
        mask = np.asfortranarray(mask.astype(np.uint8))
        if DEBUG:
            if ann['score'] < 0.1:
                continue
            print(ann['score'])
            img = (0.4 * img + 0.6 * mask.reshape(
                mask.shape[0], mask.shape[1], 1) * 255).astype(np.uint8)
            cv2.imshow('img', img)
            cv2.waitKey()
        encode = COCOmask.encode(mask)
        if 'counts' in encode:
            encode['counts'] = encode['counts'].decode("utf8")
        pred = {'image_id': ann['image_id'], 
                'category_id': ann['category_id'], 
                'score': ann['score'], 
                'segmentation': encode,
                'extreme_points': ann['extreme_points']}
        results.append(pred)
        Bar.suffix = '[{0}/{1}]| Total: {total:} | ETA: {eta:} |'.format(
            i, num_boxes, total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()
    bar.finish()
    json.dump(results, open(out_path, 'w'))
    
    dets = coco.loadRes(out_path)
    coco_eval = COCOeval(coco, dets, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
