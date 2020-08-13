from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class BYOLLoss(nn.Module):

    def __init__(self):
        super(BYOLLoss, self).__init__()

    def forward(self, online_features, target_features, labels=None, mask=None):
        if online_features.shape != target_features.shape:
            raise ValueError('Online and target features does not have the same shape')

        device = (torch.device('cuda')
                  if online_features.is_cuda
                  else torch.device('cpu'))

        if len(online_features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(online_features.shape) > 3:
            online_features = online_features.view(online_features.shape[0], online_features.shape[1], -1)
            target_features = target_features.view(target_features.shape[0], target_features.shape[1], -1)

        batch_size = online_features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = online_features.shape[1]
        online_contrast_features = torch.cat(torch.unbind(online_features, dim=1), dim=0)
        target_contrast_features = torch.cat(torch.unbind(target_features, dim=1), dim=0)


        online_contrast_features = F.normalize(online_contrast_features, dim=1)
        target_contrast_features = F.normalize(target_contrast_features, dim=1)

        distance_matrix = torch.matmul(online_contrast_features, target_contrast_features.T)

        # tile mask
        mask = mask.repeat(contrast_count, contrast_count)
        # mask-out self-contrast cases
        self_contrast_mask_out = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
            0
        )
        mask = mask * self_contrast_mask_out

        distance_matrix = distance_matrix * mask

        loss = 2 * contrast_count - 2 * distance_matrix.sum() / batch_size 

        return loss
