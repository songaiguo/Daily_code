import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes


class nms_song(object):
    def IOU_compute(self, box1, box2):

        target_upper_left_right = np.zeros([1, 4])
        for ix in range(2):
            target_upper_left_right[0, ix] = np.max([box1[ix], box2[ix]])
        for ix in range(2, 4):
            target_upper_left_right[0, ix] = np.min([box1[ix], box2[ix]])
        # Inter
        inter_width = np.max([0, target_upper_left_right[0, 2] - target_upper_left_right[0, 0]])
        inter_height = np.max([0, target_upper_left_right[0, 3] - target_upper_left_right[0, 1]])
        inter = inter_width * inter_height
        # Union
        iou = inter / ((box1[3] - box1[1]) * (box1[2] - box1[0]) + (box2[3] - box2[1]) * (box2[2] - box2[0]) - inter)
        return iou

    def nms(self, ori_boxes, iou_threshold=0.5):
        """ori_boxes: Original boxes in dimension:N*5
            iou_threshold: default 0.5"""
        boxes_tmp = np.concatenate([ori_boxes, np.reshape(np.arange(ori_boxes.shape[0]), [ori_boxes.shape[0], 1])],
                                   axis=1)
        flag = np.ones(6).astype(bool)
        flag1 = []
        while np.sum(flag) != 0:
            tmp=boxes_tmp[flag,:]
            target_box_index = np.argmax(tmp[:,-2])
            target_box_index = tmp[target_box_index,-1].astype(int)
            flag1.append(target_box_index)
            flag[target_box_index] = 0
            tmp = boxes_tmp[flag, :]
            if tmp.size != 0:
                for ix in range(tmp.shape[0]):
                    iou_tmp = self.IOU_compute(boxes_tmp[target_box_index, :], tmp[ix, :])
                    if iou_tmp >= iou_threshold:
                        flag[int(tmp[ix, -1])] = 0
        final_boxes = ori_boxes[np.array(flag1), :]
        return final_boxes

    def soft_nms(self, ori_boxes, iou_threshold=0.5):
        boxes_tmp = np.concatenate([ori_boxes, np.reshape(np.arange(ori_boxes.shape[0]), [ori_boxes.shape[0], 1])],
                                   axis=1)
        flag = np.ones(6).astype(bool)
        flag1 = []
        while np.sum(flag) != 0:
            tmp = boxes_tmp[flag, :]
            target_box_index = np.argmax(tmp[:, -2])
            target_box_index = tmp[target_box_index, -1].astype(int)
            flag1.append(target_box_index)
            flag[target_box_index] = 0
            tmp = boxes_tmp[flag, :]
            if tmp.size != 0:
                for ix in range(tmp.shape[0]):
                    iou_tmp = self.IOU_compute(boxes_tmp[target_box_index, :], tmp[ix, :])
                    if iou_tmp >= iou_threshold:
                        boxes_tmp[int(tmp[ix, -1]), -2] = boxes_tmp[int(tmp[ix, -1]), -2] * (1 - iou_tmp)
        boxes_tmp=np.delete(boxes_tmp,-1,axis=1)
        final_boxes = boxes_tmp[boxes_tmp[:,-1]>iou_threshold, :]
        return final_boxes


# Test
boxes = np.array([[100, 100, 210, 210, 0.72],
                  [250, 250, 420, 420, 0.8],
                  [220, 220, 320, 330, 0.92],
                  [100, 100, 210, 210, 0.72],
                  [230, 240, 325, 330, 0.81],
                  [100, 230, 315, 340, 0.9]])
nms_cal = nms_song()
final_box = nms_cal.nms(boxes)
print(final_box)
final_box = nms_cal.soft_nms(boxes)
print(final_box)
