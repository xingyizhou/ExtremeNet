import cv2
import math
import numpy as np
import torch
import random
import string

from config import system_configs
from utils import crop_image, normalize_, color_jittering_, lighting_
from .utils import random_crop_pts, draw_gaussian, gaussian_radius
from utils.debugger import Debugger

def _resize_image_pts(image, detections, extreme_pts, size):
    detections    = detections.copy()
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))
    
    height_ratio = new_height / height
    width_ratio  = new_width  / width
    detections[:, 0:4:2] *= width_ratio
    detections[:, 1:4:2] *= height_ratio
    extreme_pts[:, :, 0] *= width_ratio
    extreme_pts[:, :, 1] *= height_ratio
    return image, detections, extreme_pts

def _clip_detections_pts(image, detections, extreme_pts):
    detections    = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0:4:2] = np.clip(detections[:, 0:4:2], 0, width - 1)
    detections[:, 1:4:2] = np.clip(detections[:, 1:4:2], 0, height - 1)
    extreme_pts[:, :, 0] = np.clip(extreme_pts[:, :, 0], 0, width - 1)
    extreme_pts[:, :, 1] = np.clip(extreme_pts[:, :, 1], 0, height - 1)
    keep_inds  = ((detections[:, 2] - detections[:, 0]) > 0) & \
                 ((detections[:, 3] - detections[:, 1]) > 0)
    detections = detections[keep_inds]
    extreme_pts = extreme_pts[keep_inds]
    return detections, extreme_pts

def kp_detection(db, k_ind, data_aug, debug):
    data_rng   = system_configs.data_rng
    batch_size = system_configs.batch_size

    categories   = db.configs["categories"]
    input_size   = db.configs["input_size"]
    output_size  = db.configs["output_sizes"][0]

    border        = db.configs["border"]
    lighting      = db.configs["lighting"]
    rand_crop     = db.configs["rand_crop"]
    rand_color    = db.configs["rand_color"]
    rand_scales   = db.configs["rand_scales"]
    gaussian_bump = db.configs["gaussian_bump"]
    gaussian_iou  = db.configs["gaussian_iou"]
    gaussian_rad  = db.configs["gaussian_radius"]

    max_tag_len = 128

    # allocating memory
    images     = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
    t_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    l_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    b_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    r_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    ct_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    t_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    l_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    b_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    r_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    t_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    l_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    b_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    r_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    ct_tags    = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    tag_masks  = np.zeros((batch_size, max_tag_len), dtype=np.uint8)
    tag_lens   = np.zeros((batch_size, ), dtype=np.int32)

    db_size = db.db_inds.size
    for b_ind in range(batch_size):
        if not debug and k_ind == 0:
            db.shuffle_inds()

        db_ind = db.db_inds[k_ind]
        k_ind  = (k_ind + 1) % db_size

        # reading image
        image_file = db.image_file(db_ind)
        image      = cv2.imread(image_file)

        # reading detections
        detections, extreme_pts = db.detections(db_ind)

        # cropping an image randomly
        if rand_crop:
            image, detections, extreme_pts = random_crop_pts(
                image, detections, extreme_pts, 
                rand_scales, input_size, border=border)
        else:
            assert 0
            # image, detections = _full_image_crop(image, detections)

        image, detections, extreme_pts = _resize_image_pts(
            image, detections, extreme_pts, input_size)
        detections, extreme_pts = _clip_detections_pts(
            image, detections, extreme_pts)

        width_ratio  = output_size[1] / input_size[1]
        height_ratio = output_size[0] / input_size[0]

        # flipping an image randomly
        if np.random.uniform() > 0.5:
            image[:] = image[:, ::-1, :]
            width    = image.shape[1]
            detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1
            extreme_pts[:, :, 0] = width - extreme_pts[:, :, 0] - 1
            extreme_pts[:, 1, :], extreme_pts[:, 3, :] = \
                extreme_pts[:, 3, :].copy(), extreme_pts[:, 1, :].copy()
        
        image = image.astype(np.float32) / 255.
        if not debug:
            if rand_color:
                color_jittering_(data_rng, image)
                if lighting:
                    lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
        normalize_(image, db.mean, db.std)
        images[b_ind] = image.transpose((2, 0, 1))

        for ind, detection in enumerate(detections):
            category = int(detection[-1]) - 1
            extreme_pt = extreme_pts[ind]

            xt, yt = extreme_pt[0, 0], extreme_pt[0, 1]
            xl, yl = extreme_pt[1, 0], extreme_pt[1, 1]
            xb, yb = extreme_pt[2, 0], extreme_pt[2, 1]
            xr, yr = extreme_pt[3, 0], extreme_pt[3, 1]
            xct    = (xl + xr) / 2
            yct    = (yt + yb) / 2

            fxt = (xt * width_ratio)
            fyt = (yt * height_ratio)
            fxl = (xl * width_ratio)
            fyl = (yl * height_ratio)
            fxb = (xb * width_ratio)
            fyb = (yb * height_ratio)
            fxr = (xr * width_ratio)
            fyr = (yr * height_ratio)
            fxct = (xct * width_ratio)
            fyct = (yct * height_ratio)

            xt = int(fxt)
            yt = int(fyt)
            xl = int(fxl)
            yl = int(fyl)
            xb = int(fxb)
            yb = int(fyb)
            xr = int(fxr)
            yr = int(fyr)
            xct = int(fxct)
            yct = int(fyct)

            if gaussian_bump:
                width  = detection[2] - detection[0]
                height = detection[3] - detection[1]

                width  = math.ceil(width * width_ratio)
                height = math.ceil(height * height_ratio)

                if gaussian_rad == -1:
                    radius = gaussian_radius((height, width), gaussian_iou)
                    radius = max(0, int(radius))
                else:
                    radius = gaussian_rad
                draw_gaussian(t_heatmaps[b_ind, category], [xt, yt], radius)
                draw_gaussian(l_heatmaps[b_ind, category], [xl, yl], radius)
                draw_gaussian(b_heatmaps[b_ind, category], [xb, yb], radius)
                draw_gaussian(r_heatmaps[b_ind, category], [xr, yr], radius)
                draw_gaussian(ct_heatmaps[b_ind, category], [xct, yct], radius)
            else:
                t_heatmaps[b_ind, category, yt, xt] = 1
                l_heatmaps[b_ind, category, yl, xl] = 1
                b_heatmaps[b_ind, category, yb, xb] = 1
                r_heatmaps[b_ind, category, yr, xr] = 1

            tag_ind = tag_lens[b_ind]
            t_regrs[b_ind, tag_ind, :] = [fxt - xt, fyt - yt]
            l_regrs[b_ind, tag_ind, :] = [fxl - xl, fyl - yl]
            b_regrs[b_ind, tag_ind, :] = [fxb - xb, fyb - yb]
            r_regrs[b_ind, tag_ind, :] = [fxr - xr, fyr - yr]
            t_tags[b_ind, tag_ind] = yt * output_size[1] + xt
            l_tags[b_ind, tag_ind] = yl * output_size[1] + xl
            b_tags[b_ind, tag_ind] = yb * output_size[1] + xb
            r_tags[b_ind, tag_ind] = yr * output_size[1] + xr
            ct_tags[b_ind, tag_ind] = yct * output_size[1] + xct
            tag_lens[b_ind] += 1

    for b_ind in range(batch_size):
        tag_len = tag_lens[b_ind]
        tag_masks[b_ind, :tag_len] = 1

    if debug:
        debugger = Debugger(num_classes=80)
        t_hm = debugger.gen_colormap(t_heatmaps[0])
        l_hm = debugger.gen_colormap(l_heatmaps[0])
        b_hm = debugger.gen_colormap(b_heatmaps[0])
        r_hm = debugger.gen_colormap(r_heatmaps[0])
        ct_hm = debugger.gen_colormap(ct_heatmaps[0])
        img = images[0] * db.std.reshape(3, 1, 1) + db.mean.reshape(3, 1, 1)
        img =  (img * 255).astype(np.uint8).transpose(1, 2, 0)
        debugger.add_blend_img(img, t_hm, 't_hm')
        debugger.add_blend_img(img, l_hm, 'l_hm')
        debugger.add_blend_img(img, b_hm, 'b_hm')
        debugger.add_blend_img(img, r_hm, 'r_hm')
        debugger.add_blend_img(
            img, np.maximum(np.maximum(t_hm, l_hm), 
                            np.maximum(b_hm, r_hm)), 'extreme')
        debugger.add_blend_img(img, ct_hm, 'center')
        debugger.show_all_imgs(pause=True)

    images     = torch.from_numpy(images)
    t_heatmaps = torch.from_numpy(t_heatmaps)
    l_heatmaps = torch.from_numpy(l_heatmaps)
    b_heatmaps = torch.from_numpy(b_heatmaps)
    r_heatmaps = torch.from_numpy(r_heatmaps)
    ct_heatmaps = torch.from_numpy(ct_heatmaps)
    t_regrs    = torch.from_numpy(t_regrs)
    l_regrs    = torch.from_numpy(l_regrs)
    b_regrs    = torch.from_numpy(b_regrs)
    r_regrs    = torch.from_numpy(r_regrs)
    t_tags     = torch.from_numpy(t_tags)
    l_tags     = torch.from_numpy(l_tags)
    b_tags     = torch.from_numpy(b_tags)
    r_tags     = torch.from_numpy(r_tags)
    ct_tags    = torch.from_numpy(ct_tags)
    tag_masks  = torch.from_numpy(tag_masks)

    return {
        "xs": [images, t_tags, l_tags, b_tags, r_tags, ct_tags],
        "ys": [t_heatmaps, l_heatmaps, b_heatmaps, r_heatmaps, ct_heatmaps,
               tag_masks, t_regrs, l_regrs, b_regrs, r_regrs]
    }, k_ind

def sample_data(db, k_ind, data_aug=True, debug=False):
    return globals()[system_configs.sampling_function](db, k_ind, data_aug, debug)
