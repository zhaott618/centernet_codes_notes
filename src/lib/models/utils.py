from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    ###对tensor进行某种维度上的gathers，在指定维度上按照索引赋值输出tensor。输入与输出大小一致

    feat = feat.gather(1, ind)
    if mask is not None:
        ###去除所有尺度为1的维度
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        ###feat的shape变为（-1，dim）
        feat = feat.view(-1, dim)
    return feat

####这里是通过topk_inds(注意不是topk_ind)来找到网络输出的wh, reg, hm分支中的对应的k个最大值
####topk_inds也是通过topk_ind通过gather()得到的，gather过程将topk_inds的shape由(batch, cat, k)
####变为(batch, k), 但留下的k个索引还是原来的索引(可以判断出cls的索引)
def _tranpose_and_gather_feat(feat, ind):
	####例如wh(即heat)原尺寸为(batch, channel, width, height)变为(b, w, h, c)
    feat = feat.permute(0, 2, 3, 1).contiguous()
	####将feat的shape变为(b, w*h, c)
    feat = feat.view(feat.size(0), -1, feat.size(3))
	####根据ind筛选出feat中对应的元素, ind的size为(b, k, 1)
	####feat的shape变为(b, k, 1)
    feat = _gather_feat(feat, ind)
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)