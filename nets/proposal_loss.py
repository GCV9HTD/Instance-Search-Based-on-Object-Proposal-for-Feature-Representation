import torch
import torch.nn as nn
import numpy as np
import math

from common.utils import bbox_iou


class ProposalLoss(nn.Module):
    def __init__(self, anchors, img_size):
        """

        :param anchors:
        :param num_classes:
        :param img_size:
        """
        super(ProposalLoss, self).__init__()

        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.bbox_attrs = 5
        self.img_size = img_size

        self.ignore_threshold = 0.5  # IoU threshold
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_noobj = 0.5
        self.lambda_obj = 1.0

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()


    def forward(self, input, targets=None):
        """

        :param input:
        :param targets:
        :return:
        """
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        prediction = input.view(bs, self.num_anchors, self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])

        if targets is not None:
            # build target
            obj_mask, noobj_mask, tx, ty, tw, th = self.get_target(targets, scaled_anchors, in_w, in_h, self.ignore_threshold)
            obj_mask, noobj_mask = obj_mask.cuda(), noobj_mask.cuda()
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()

            # losses
            loss_x = self.bce_loss(x * obj_mask, tx * obj_mask)
            loss_y = self.bce_loss(y * obj_mask, ty * obj_mask)
            loss_w = self.mse_loss(w * obj_mask, tw * obj_mask)
            loss_h = self.mse_loss(h * obj_mask, th * obj_mask)
            loss_conf = self.lambda_obj * self.bce_loss(conf * obj_mask, 1.0 * obj_mask) + \
                        self.lambda_noobj * self.bce_loss(conf * noobj_mask, 0.0 * noobj_mask)
            loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                   loss_w * self.lambda_wh + loss_h * self.lambda_wh + loss_conf * self.lambda_conf

            return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item()
        else:
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensorBase

            # Calculate offsets for each grid
            grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(
                bs * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(
                bs * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

            # Calculate anchor w, h
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
            anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

            # Add offset and scale with anchors
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

            # Results
            _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
            output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale, conf.view(bs, -1, 1)), -1)

            return output.data


    def get_target(self, target, anchors, in_w, in_h, ignore_threshold):
        """

        :param target:
        :param anchors:
        :param in_w:
        :param in_h:
        :param ignore_threshold:
        :return:
        """
        bs = target.size(0)

        obj_mask       = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, self.num_anchors, in_h, in_w, requires_grad=False)

        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)

        for b in range(bs):
            for t in range(target.shape[1]):

                if target[b, t].sum() == 0:
                    continue

                # Convert to position relative to box
                gx = target[b, t, 0] * in_w
                gy = target[b, t, 1] * in_h
                gw = target[b, t, 2] * in_w
                gh = target[b, t, 3] * in_h

                # Get grid box indices
                gi = int(gx)
                gj = int(gy)

                # Get shape of gt box
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)

                # Get shape of anchor box, [0, 0, w, h] style
                anchor_shapes = torch.FloatTensor(
                    np.concatenate((np.zeros((self.num_anchors, 2)), np.array(anchors)), 1)
                )

                # Calculate iou between gt and anchor shapes
                anch_ious = bbox_iou(gt_box, anchor_shapes)

                # If the overlap is larger than threshold, set no-object mask to zero (ignore)
                # gj before gi
                noobj_mask[b, anch_ious > ignore_threshold, gj, gi] = 0

                # Find the best matching anchor box
                best_n = np.argmax(anch_ious)

                # Object mask
                obj_mask[b, best_n, gj, gi] = 1

                # Coordinates, Cx and Cy
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj

                # Width and height
                tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)

        return obj_mask, noobj_mask, tx, ty, tw, th
