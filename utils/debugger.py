import numpy as np
import cv2
import matplotlib.pyplot as plt

color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255
  
def show_2d(img, points, c, edges):
  num_joints = points.shape[0]
  points = ((points.reshape(num_joints, -1))).astype(np.int32)
  for j in range(num_joints):
    cv2.circle(img, (points[j, 0], points[j, 1]), 3, c, -1)
  for e in edges:
    if points[e].min() > 0:
      cv2.line(img, (points[e[0], 0], points[e[0], 1]),
                    (points[e[1], 0], points[e[1], 1]), c, 2)
  return img

class Debugger(object):
  def __init__(self, ipynb = False, num_classes=80):
    self.ipynb = ipynb
    if not self.ipynb:
      self.plt = plt
      self.fig = self.plt.figure()
    self.imgs = {}
    # colors = [((np.random.random((3, )) * 0.6 + 0.4)*255).astype(np.uint8) \
    #           for _ in range(num_classes)]
    colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(num_classes)]
    self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)

  def add_img(self, img, imgId = 'default', revert_color=False):
    if revert_color:
      img = 255 - img
    self.imgs[imgId] = img.copy()
  
  def add_mask(self, mask, bg, imgId = 'default', trans = 0.8):
    self.imgs[imgId] = (mask.reshape(mask.shape[0], mask.shape[1], 1) * 255 * trans + \
                        bg * (1 - trans)).astype(np.uint8)

  def add_point_2d(self, point, c, edges, imgId = 'default'):
    self.imgs[imgId] = show_2d(self.imgs[imgId], point, c, edges)
  
  def show_img(self, pause = False, imgId = 'default'):
    cv2.imshow('{}'.format(imgId), self.imgs[imgId])
    if pause:
      cv2.waitKey()
  
  def add_blend_img(self, back, fore, imgId='blend', trans=0.5):
    # fore = 255 - fore
    if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
      fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
    if len(fore.shape) == 2:
      fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
    self.imgs[imgId] = (back * (1. - trans) + fore * trans)
    self.imgs[imgId][self.imgs[imgId] > 255] = 255
    self.imgs[imgId] = self.imgs[imgId].astype(np.uint8)

  def gen_colormap(self, img, s=4):
    num_classes = len(self.colors)
    img[img < 0] = 0
    h, w = img.shape[1], img.shape[2]
    color_map = np.zeros((h*s, w*s, 3), dtype=np.uint8)
    for i in range(num_classes):
      resized = cv2.resize(img[i], (w*s, h*s)).reshape(h*s, w*s, 1)
      cl =  self.colors[i]
      color_map = np.maximum(color_map, (resized * cl).astype(np.uint8))
    return color_map

  def add_rect(self, rect1, rect2, c, conf=1, imgId = 'default'): 
    cv2.rectangle(self.imgs[imgId], (rect1[0], rect1[1]), (rect2[0], rect2[1]), c, 2)
    if conf < 1:
      cv2.circle(self.imgs[imgId], (rect1[0], rect1[1]), int(10 * conf), c, 1)
      cv2.circle(self.imgs[imgId], (rect2[0], rect2[1]), int(10 * conf), c, 1)
      cv2.circle(self.imgs[imgId], (rect1[0], rect2[1]), int(10 * conf), c, 1)
      cv2.circle(self.imgs[imgId], (rect2[0], rect1[1]), int(10 * conf), c, 1)

  def add_points(self, points, img_id = 'default'):
    num_classes = len(points)
    assert num_classes == len(self.colors)
    for i in range(num_classes):
      for j in range(len(points[i])):
        c = self.colors[i, 0, 0]
        cv2.circle(self.imgs[img_id], (points[i][j][0] * 4, points[i][j][1] * 4),
                   5, (255, 255, 255), -1)
        cv2.circle(self.imgs[img_id], (points[i][j][0] * 4, points[i][j][1] * 4),
                   3, (int(c[0]), int(c[1]), int(c[2])), -1)

  def show_all_imgs(self, pause=False):
    if not self.ipynb:
      for i, v in self.imgs.items():
        cv2.imshow('{}'.format(i), v)
      if pause:
        cv2.waitKey()
    else:
      self.ax = None
      nImgs = len(self.imgs)
      fig=plt.figure(figsize=(nImgs * 10,10))
      nCols = nImgs
      nRows = nImgs // nCols
      for i, (k, v) in enumerate(self.imgs.items()):
        fig.add_subplot(1, nImgs, i + 1)
        if len(v.shape) == 3:
          plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        else:
          plt.imshow(v)
      plt.show()

  def save_img(self, imgId='default', path='./cache/debug/'):
    cv2.imwrite(path + '{}.png'.format(imgId), self.imgs[imgId])
    
  def save_all_imgs(self, path='./cache/debug/', prefix='', genID=False):
    if genID:
      try:
        idx = int(np.loadtxt(path + '/id.txt'))
      except:
        idx = 0
      prefix=idx
      np.savetxt(path + '/id.txt', np.ones(1) * (idx + 1), fmt='%d')
    for i, v in self.imgs.items():
      cv2.imwrite(path + '/{}{}.png'.format(prefix, i), v)
    
