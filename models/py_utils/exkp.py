import numpy as np
import torch
import torch.nn as nn

from .utils import convolution, residual
from .utils import make_layer, make_layer_revr

from .kp_utils import _tranpose_and_gather_feat, _exct_decode
from .kp_utils import _sigmoid, _regr_loss, _neg_loss
from .kp_utils import make_kp_layer
from .kp_utils import make_pool_layer, make_unpool_layer
from .kp_utils import make_merge_layer, make_inter_layer, make_cnv_layer
from .kp_utils import _h_aggregate, _v_aggregate
from utils.debugger import Debugger

class kp_module(nn.Module):
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n   = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod, 
            layer=layer, **kwargs
        )  
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer, 
            make_up_layer=make_up_layer, 
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs
        )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2  = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return self.merge(up1, up2)

class exkp(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256, 
        make_tl_layer=None, make_br_layer=None,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer, 
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer, 
        kp_layer=residual
    ):
        super(exkp, self).__init__()

        self.nstack    = nstack
        self._decode   = _exct_decode

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.t_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        self.l_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        self.b_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        self.r_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        self.ct_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        for t_heat, l_heat, b_heat, r_heat, ct_heat in \
          zip(self.t_heats, self.l_heats, self.b_heats, \
              self.r_heats, self.ct_heats):
            t_heat[-1].bias.data.fill_(-2.19)
            l_heat[-1].bias.data.fill_(-2.19)
            b_heat[-1].bias.data.fill_(-2.19)
            r_heat[-1].bias.data.fill_(-2.19)
            ct_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.t_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.l_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.b_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.r_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)

    def _train(self, *xs):
        image  = xs[0]
        t_inds = xs[1]
        l_inds = xs[2]
        b_inds = xs[3]
        r_inds = xs[4]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.t_heats, self.l_heats, self.b_heats, self.r_heats,
            self.ct_heats,
            self.t_regrs, self.l_regrs, self.b_regrs, self.r_regrs,
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            t_heat_, l_heat_, b_heat_, r_heat_ = layer[2:6]
            ct_heat_                           = layer[6]
            t_regr_, l_regr_, b_regr_, r_regr_ = layer[7:11]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            t_heat, l_heat = t_heat_(cnv), l_heat_(cnv)
            b_heat, r_heat = b_heat_(cnv), r_heat_(cnv)
            ct_heat        = ct_heat_(cnv)

            t_regr, l_regr = t_regr_(cnv), l_regr_(cnv)
            b_regr, r_regr = b_regr_(cnv), r_regr_(cnv)

            t_regr = _tranpose_and_gather_feat(t_regr, t_inds)
            l_regr = _tranpose_and_gather_feat(l_regr, l_inds)
            b_regr = _tranpose_and_gather_feat(b_regr, b_inds)
            r_regr = _tranpose_and_gather_feat(r_regr, r_inds)

            outs += [t_heat, l_heat, b_heat, r_heat, ct_heat, \
                     t_regr, l_regr, b_regr, r_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.t_heats, self.l_heats, self.b_heats, self.r_heats,
            self.ct_heats,
            self.t_regrs, self.l_regrs, self.b_regrs, self.r_regrs,
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_                          = layer[0:2]
            t_heat_, l_heat_, b_heat_, r_heat_ = layer[2:6]
            ct_heat_                           = layer[6]
            t_regr_, l_regr_, b_regr_, r_regr_ = layer[7:11]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.nstack - 1:
                t_heat, l_heat = t_heat_(cnv), l_heat_(cnv)
                b_heat, r_heat = b_heat_(cnv), r_heat_(cnv)
                ct_heat        = ct_heat_(cnv)

                t_regr, l_regr = t_regr_(cnv), l_regr_(cnv)
                b_regr, r_regr = b_regr_(cnv), r_regr_(cnv)

                outs += [t_heat, l_heat, b_heat, r_heat, ct_heat,
                         t_regr, l_regr, b_regr, r_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        if kwargs['debug']:
            _debug(image, t_heat, l_heat, b_heat, r_heat, ct_heat)
        del kwargs['debug']
        return self._decode(*outs[-9:], **kwargs)

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

class CTLoss(nn.Module):
    def __init__(self, regr_weight=1, focal_loss=_neg_loss):
        super(CTLoss, self).__init__()

        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.regr_loss   = _regr_loss

    def forward(self, outs, targets):
        stride = 9

        t_heats  = outs[0::stride]
        l_heats  = outs[1::stride]
        b_heats  = outs[2::stride]
        r_heats  = outs[3::stride]
        ct_heats = outs[4::stride]
        t_regrs  = outs[5::stride]
        l_regrs  = outs[6::stride]
        b_regrs  = outs[7::stride]
        r_regrs  = outs[8::stride]

        gt_t_heat  = targets[0]
        gt_l_heat  = targets[1]
        gt_b_heat  = targets[2]
        gt_r_heat  = targets[3]
        gt_ct_heat = targets[4]
        gt_mask    = targets[5]
        gt_t_regr  = targets[6]
        gt_l_regr  = targets[7]
        gt_b_regr  = targets[8]
        gt_r_regr  = targets[9]

        # focal loss
        focal_loss = 0

        t_heats  = [_sigmoid(t) for t in t_heats]
        l_heats  = [_sigmoid(l) for l in l_heats]
        b_heats  = [_sigmoid(b) for b in b_heats]
        r_heats  = [_sigmoid(r) for r in r_heats]
        ct_heats = [_sigmoid(ct) for ct in ct_heats]

        focal_loss += self.focal_loss(t_heats, gt_t_heat)
        focal_loss += self.focal_loss(l_heats, gt_l_heat)
        focal_loss += self.focal_loss(b_heats, gt_b_heat)
        focal_loss += self.focal_loss(r_heats, gt_r_heat)
        focal_loss += self.focal_loss(ct_heats, gt_ct_heat)

        # regression loss
        regr_loss = 0
        for t_regr, l_regr, b_regr, r_regr in \
          zip(t_regrs, l_regrs, b_regrs, r_regrs):
            regr_loss += self.regr_loss(t_regr, gt_t_regr, gt_mask)
            regr_loss += self.regr_loss(l_regr, gt_l_regr, gt_mask)
            regr_loss += self.regr_loss(b_regr, gt_b_regr, gt_mask)
            regr_loss += self.regr_loss(r_regr, gt_r_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss

        loss = (focal_loss + regr_loss) / len(t_heats)
        return loss.unsqueeze(0)

def _debug(image, t_heat, l_heat, b_heat, r_heat, ct_heat):
    debugger = Debugger(num_classes=80)
    k = 0

    t_heat = torch.sigmoid(t_heat)
    l_heat = torch.sigmoid(l_heat)
    b_heat = torch.sigmoid(b_heat)
    r_heat = torch.sigmoid(r_heat)
    
    
    aggr_weight = 0.1
    t_heat = _h_aggregate(t_heat, aggr_weight=aggr_weight)
    l_heat = _v_aggregate(l_heat, aggr_weight=aggr_weight)
    b_heat = _h_aggregate(b_heat, aggr_weight=aggr_weight)
    r_heat = _v_aggregate(r_heat, aggr_weight=aggr_weight)
    t_heat[t_heat > 1] = 1
    l_heat[l_heat > 1] = 1
    b_heat[b_heat > 1] = 1
    r_heat[r_heat > 1] = 1
    
    
    ct_heat = torch.sigmoid(ct_heat)

    t_hm = debugger.gen_colormap(t_heat[k].cpu().data.numpy())
    l_hm = debugger.gen_colormap(l_heat[k].cpu().data.numpy())
    b_hm = debugger.gen_colormap(b_heat[k].cpu().data.numpy())
    r_hm = debugger.gen_colormap(r_heat[k].cpu().data.numpy())
    ct_hm = debugger.gen_colormap(ct_heat[k].cpu().data.numpy())

    hms = np.maximum(np.maximum(t_hm, l_hm), 
                     np.maximum(b_hm, r_hm))
    # debugger.add_img(hms, 'hms')
    if image is not None:
        mean = np.array([0.40789654, 0.44719302, 0.47026115],
                        dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.28863828, 0.27408164, 0.27809835],
                        dtype=np.float32).reshape(3, 1, 1)
        img = (image[k].cpu().data.numpy() * std + mean) * 255
        img = img.astype(np.uint8).transpose(1, 2, 0)
        debugger.add_img(img, 'img')
        # debugger.add_blend_img(img, t_hm, 't_hm')
        # debugger.add_blend_img(img, l_hm, 'l_hm')
        # debugger.add_blend_img(img, b_hm, 'b_hm')
        # debugger.add_blend_img(img, r_hm, 'r_hm')
        debugger.add_blend_img(img, hms, 'extreme')
        debugger.add_blend_img(img, ct_hm, 'center')
    debugger.show_all_imgs(pause=False)
