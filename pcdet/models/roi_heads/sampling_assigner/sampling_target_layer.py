import numpy as np
import torch
import torch.nn as nn

from ....ops.iou3d_nms import iou3d_nms_utils

class SamplingTargetLayer(nn.Module):
    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg

    def limit(self,ang):
        ang = ang % (2 * np.pi)

        ang[ang > np.pi] = ang[ang > np.pi] - 2 * np.pi

        ang[ang < -np.pi] = ang[ang < -np.pi] + 2 * np.pi

        return ang

    def ang_weight(self,pred, gt):

        a = torch.abs(pred - gt)
        b = 2 * np.pi - torch.abs(pred - gt)

        res = torch.stack([a, b])

        res = torch.min(res, 0)[0]

        return 1 - res / np.pi
    
    def generate_sampling_roi(self, batch_dict, i, enable_dif = False):
        '''
        Args:
            gt_boxes: (N, 8)
            sampling_num: 100,
            offset_threshold:[0.0, 0.5],
            offset_split: 10
        Return:
            sampling_rois,
            sampling_rois_labels
        '''
        torch.manual_seed(0)
        if i==0 or enable_dif==False:
            str_idx = ''
        else:
            str_idx = str(i)
        
        cur_cfg = self.roi_sampler_cfg

        batch_gt_boxes = batch_dict['gt_boxes'+str_idx]
        num_sample = cur_cfg.SAM_PER_IMAGE
        num_radom = cur_cfg.RANDOM_NUM
        sampling_each_bbox = cur_cfg.SAMPLING_EACH_BBOX

        offset_threshold = cur_cfg.OFFSET_THRESHOLD
        scale_threshold = cur_cfg.SCALE_THRESHOLD
        iou_threshold = cur_cfg.IOU_THRESHOLD
        batch_generate_rois = batch_gt_boxes.new_zeros((batch_gt_boxes.shape[0],num_sample,batch_gt_boxes.shape[-1]-1))
        batch_generate_rois_labels = batch_gt_boxes.new_zeros((batch_gt_boxes.shape[0],num_sample)).to(torch.int64)

        for j in range(batch_gt_boxes.shape[0]):
            gt_boxes = batch_gt_boxes[j]
            k = gt_boxes.__len__() - 1
            while k > 0 and gt_boxes[k].sum() == 0:
                k -= 1
            gt_boxes = gt_boxes[:k + 1]
            new_boxes_list, new_ious_list, new_boxes_labels_list=[], [], []

            for i in range(gt_boxes.shape[0]):
                
                gt_box = gt_boxes[i].unsqueeze(0)

                cxcycz = gt_box[:,:3]
                whd = gt_box[:,3:6]
                dir = gt_box[:,6]

                for split_idx in range(len(offset_threshold)):
                    xyz_offset = torch.empty((num_radom,3), device=gt_box.device, dtype=gt_box.dtype).uniform_(-offset_threshold[split_idx],offset_threshold[split_idx])
                    whd_offset = torch.empty((num_radom,3), device=gt_box.device, dtype=gt_box.dtype).uniform_(scale_threshold[split_idx], 1./scale_threshold[split_idx])
                    dir_offset = torch.empty((num_radom,1), device=gt_box.device, dtype=gt_box.dtype).uniform_(-offset_threshold[split_idx],offset_threshold[split_idx]) * torch.pi/2
                    new_xyz = cxcycz + whd * xyz_offset
                    new_whd = whd * whd_offset
                    new_dir = dir + dir_offset
                    new_boxes = torch.cat([new_xyz,new_whd,new_dir],dim=-1)

                    iou3d = iou3d_nms_utils.boxes_iou3d_gpu(new_boxes[:,0:7], gt_boxes[:,0:7]) # (num_random , N)
                    max_iou, index = iou3d.max(dim=1)
                    mask_idx = (max_iou > iou_threshold[split_idx][0]) & (max_iou <= iou_threshold[split_idx][1]) & (index == i)
                    new_boxes = new_boxes[mask_idx,:]
                    new_boxes_labels = gt_box[:,-1].repeat(new_boxes.shape[0],1)
                    if new_boxes.shape[0] > sampling_each_bbox:
                        new_boxes = new_boxes[:sampling_each_bbox]
                        new_boxes_labels = new_boxes_labels[:sampling_each_bbox]
                    new_boxes_list.append(new_boxes.reshape(-1,7))
                    new_boxes_labels_list.append(new_boxes_labels.reshape(-1,1))
            
            cur_generate_rois = torch.cat(new_boxes_list, dim=0)
            cur_generate_rois_labels = torch.cat(new_boxes_labels_list, dim=0)
            total_cnt = cur_generate_rois.shape[0]
            assert total_cnt >= num_sample, "THE SAMPLING BBOXES CAN NOT FIX THE NEEDED"
            if total_cnt >= num_sample:
                cands = np.arange(total_cnt)
                np.random.shuffle(cands)
                final_idx = cands[:num_sample]
                final_idx = torch.from_numpy(final_idx).long().to(cur_generate_rois.device)
                cur_generate_rois = cur_generate_rois[final_idx,:]
                cur_generate_rois_labels = cur_generate_rois_labels[final_idx,:]

            batch_generate_rois[j] = cur_generate_rois
            batch_generate_rois_labels[j] = cur_generate_rois_labels.squeeze(-1).to(torch.int64)

        return batch_generate_rois, batch_generate_rois_labels

    def forward(self, batch_dict, ind=''):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
            batch_dict:
                rois: (B, M, 7 + C)
                gt_of_rois: (B, M, 7 + C)
                gt_iou_of_rois: (B, M)
                roi_scores: (B, M)
                roi_labels: (B, M)
                reg_valid_mask: (B, M)
                rcnn_cls_labels: (B, M)
        """
        batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_labels = self.sample_rois_for_rcnn(batch_dict=batch_dict, ind=ind)
        # regression valid mask


        if self.roi_sampler_cfg.CLS_SCORE_TYPE in ['roi_iou_x', 'roi_ioud_x']:
            reg_valid_mask = batch_roi_ious.new_zeros(batch_roi_ious.shape).long()
            for cls_i in range(len(self.roi_sampler_cfg.REG_FG_THRESH)):
                reg_fg_thresh = self.roi_sampler_cfg.REG_FG_THRESH[cls_i]
                cls_mask = batch_gt_of_rois[...,-1] == (cls_i+1)


                if self.roi_sampler_cfg.get('ENABLE_HARD_SAMPLING', False):
                    mask_hard = (batch_roi_ious < reg_fg_thresh) & (batch_roi_ious > self.roi_sampler_cfg.HARD_SAMPLING_THRESH[cls_i]) & cls_mask

                    mask_prob = mask_hard.new_zeros(mask_hard.size()).bool()
                    teval = int(1/self.roi_sampler_cfg.HARD_SAMPLING_RATIO[cls_i])
                    ints = range(np.random.randint(0, teval), mask_prob.shape[0], teval)

                    mask_prob[ints] = 1

                    mask_hard2 = mask_hard * mask_prob

                    this_fg_inds1 = ((batch_roi_ious > reg_fg_thresh) & cls_mask).long()
                    this_reg_valid_mask = this_fg_inds1 + mask_hard2.long()

                else:
                    this_reg_valid_mask = ((batch_roi_ious > reg_fg_thresh) & cls_mask).long()
                reg_valid_mask += this_reg_valid_mask
        else:
            reg_valid_mask = (batch_roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long()

        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious,
                        'roi_labels': batch_roi_labels,
                        'reg_valid_mask': reg_valid_mask}

        return targets_dict
    



    def sample_rois_for_rcnn(self, batch_dict, ind=''):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['sampling_rois']
        roi_labels = batch_dict['sampling_rois_labels']
        gt_boxes = batch_dict['gt_boxes'+ind]

        sampling_roi_num = rois.shape[1]

        gt_code_size = gt_boxes.shape[-1]
        roi_code_size = rois.shape[-1]

        batch_sam_rois = rois.new_zeros(batch_size, sampling_roi_num, roi_code_size)
        batch_sam_gt_of_rois = rois.new_zeros(batch_size, sampling_roi_num, gt_code_size )
        batch_sam_rois_ious = rois.new_zeros(batch_size, sampling_roi_num)
        batch_sam_rois_labels = rois.new_zeros(batch_size, sampling_roi_num)


        for index in range(batch_size):
            cur_roi, cur_gt, cur_roi_labels = rois[index], gt_boxes[index], roi_labels[index]
            k = cur_gt.__len__() - 1
            while k > 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt

            if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                    rois=cur_roi, roi_labels=cur_roi_labels,
                    gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long()
                )
            else:
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

            if self.roi_sampler_cfg.CLS_SCORE_TYPE in ['roi_iou_x','roi_ioud_x']:
                sampled_inds = self.subsample_rois(max_overlaps=max_overlaps, gts = cur_gt[gt_assignment])
            else:
                sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)

            batch_sam_rois[index] = cur_roi[sampled_inds]
            batch_sam_rois_labels[index] = cur_roi_labels[sampled_inds]
            batch_sam_rois_ious[index] = max_overlaps[sampled_inds]
            
            batch_sam_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]

        return batch_sam_rois, batch_sam_gt_of_rois, batch_sam_rois_ious, batch_sam_rois_labels

    def subsample_rois(self, max_overlaps, gts=None):
        # sample fg, easy_bg, hard_bg
        sample_num = len(max_overlaps)
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * (sample_num)))

        if gts is None:
            fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH)
            fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)
        else:
            fg_inds = max_overlaps.new_zeros(max_overlaps.shape).long()
            for i in range(len(self.roi_sampler_cfg.CLS_FG_THRESH)):
                cls_mask = gts[...,-1] == (i+1)
                this_fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH[i], self.roi_sampler_cfg.CLS_FG_THRESH[i])

                this_fg_inds = ((max_overlaps >= this_fg_thresh) & cls_mask)

                fg_inds+=this_fg_inds
            fg_inds = fg_inds.nonzero().view(-1)


        easy_bg_inds = ((max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)

        if gts is None:
            hard_bg_inds = ((max_overlaps < self.roi_sampler_cfg.REG_FG_THRESH) &
                            (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)
        else:
            hard_bg_inds = max_overlaps.new_zeros(max_overlaps.shape).long()
            for i in range(len(self.roi_sampler_cfg.REG_FG_THRESH)):
                cls_mask = gts[...,-1] == (i+1)
                this_hard_bg_inds = ((max_overlaps < self.roi_sampler_cfg.REG_FG_THRESH[i]) &
                                (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO) & cls_mask)
                hard_bg_inds+=this_hard_bg_inds
            hard_bg_inds = hard_bg_inds.nonzero().view(-1)


        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = sample_num - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
            sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(sample_num) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = []
            sampled_inds = fg_inds

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = sample_num
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
            sampled_inds = bg_inds
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError
        
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (N, )
            gt_labels:

        Returns:

        """
        """
        :param rois: (N, 7)
        :param roi_labels: (N)
        :param gt_boxes: (N, 8)
        :return:
        """
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
            roi_mask = (roi_labels == k)
            gt_mask = (gt_labels == k)
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                original_gt_assignment = gt_mask.nonzero().view(-1)

                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi[:,0:7], cur_gt[:,0:7])  # (M, N)
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                max_overlaps[roi_mask] = cur_max_overlaps
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

        return max_overlaps, gt_assignment
