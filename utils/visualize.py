import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pycocotools.mask as mask_util

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)

def vis_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=2):
    """Visualizes a single binary mask."""

    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col

    if show_border:
        # How to use `cv2.findContours` in different OpenCV versions?
        # https://stackoverflow.com/questions/48291581/how-to-use-cv2-findcontours-in-different-opencv-versions/48292371#48292371
        contours = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[-2]
        cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)


def vis_octagon(img, extreme_points, col, border_thick=2):
    """Visualizes a single binary mask."""

    img = img.astype(np.uint8)
    # COL = (col).astype(np.uint8).tolist()
    # print('col', COL)
    # octagon = get_octagon(extreme_points)
    # octagon = np.array(octagon).reshape(8, 1, 2).astype(np.int32)
    # cv2.polylines(img, [octagon], 
    #               True, COL, border_thick)
    mask = extreme_point_to_octagon_mask(
      extreme_points, img.shape[0], img.shape[1])

    img = vis_mask(img, mask, col)

    return img.astype(np.uint8)

def vis_ex(img, extreme_points, col, border_thick=2):
    """Visualizes a single binary mask."""

    img = img.astype(np.uint8)
    COL = (col).astype(np.uint8).tolist()
    # print('col', COL)
    ex = np.array(extreme_points).reshape(4, 2).astype(np.int32)
    
    L = 10
    T = 0.7
    cv2.arrowedLine(img, (ex[0][0], ex[0][1] + L), (ex[0][0], ex[0][1]), COL, border_thick, tipLength=T)
    cv2.arrowedLine(img, (ex[1][0] + L, ex[1][1]), (ex[1][0], ex[1][1]), COL, border_thick, tipLength=T)
    cv2.arrowedLine(img, (ex[2][0], ex[2][1] - L), (ex[2][0], ex[2][1]), COL, border_thick, tipLength=T)
    cv2.arrowedLine(img, (ex[3][0] - L, ex[3][1]), (ex[3][0], ex[3][1]), COL, border_thick, tipLength=T)
    
    '''
    R = 6
    cv2.circle(img, (ex[0][0], ex[0][1]), R, COL, -1)
    cv2.circle(img, (ex[1][0], ex[1][1]), R, COL, -1)
    cv2.circle(img, (ex[2][0], ex[2][1]), R, COL, -1)
    cv2.circle(img, (ex[3][0], ex[3][1]), R, COL, -1)

    cv2.circle(img, (ex[0][0], ex[0][1]), R, _WHITE, 2)
    cv2.circle(img, (ex[1][0], ex[1][1]), R, _WHITE, 2)
    cv2.circle(img, (ex[2][0], ex[2][1]), R, _WHITE, 2)
    cv2.circle(img, (ex[3][0], ex[3][1]), R, _WHITE, 2)
    '''
    return img.astype(np.uint8)


def vis_class(img, pos, class_str, font_scale=0.35):
    """Visualizes the class."""
    img = img.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    if y0 - int(1.3 * txt_h) < 0:
      y0 = y0 + int(1.6 * txt_h)
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, _GREEN, -1)
    # cv2.rectangle(img, back_tl, back_br, _GRAY, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    # cv2.putText(img, txt, txt_tl, font, font_scale, (46, 52, 54), lineType=cv2.LINE_AA)
    return img


def vis_bbox(img, bbox, thick=2):
    """Visualizes a bounding box."""
    img = img.astype(np.uint8)
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), _GREEN, thickness=thick)
    return img

def get_octagon(ex):
  ex = np.array(ex).reshape(4, 2)
  w, h = ex[3][0] - ex[1][0], ex[2][1] - ex[0][1]
  t, l, b, r = ex[0][1], ex[1][0], ex[2][1], ex[3][0]
  x = 8.
  octagon = [[min(ex[0][0] + w / x, r), ex[0][1], \
              max(ex[0][0] - w / x, l), ex[0][1], \
              ex[1][0], max(ex[1][1] - h / x, t), \
              ex[1][0], min(ex[1][1] + h / x, b), \
              max(ex[2][0] - w / x, l), ex[2][1], \
              min(ex[2][0] + w / x, r), ex[2][1], \
              ex[3][0], min(ex[3][1] + h / x, b), \
              ex[3][0], max(ex[3][1] - h / x, t)
              ]]
  return octagon

def extreme_point_to_octagon_mask(extreme_points, h, w):
  octagon = get_octagon(extreme_points)
  rles = mask_util.frPyObjects(octagon, h, w)
  rle = mask_util.merge(rles)
  mask = mask_util.decode(rle)
  return mask
