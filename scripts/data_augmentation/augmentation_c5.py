# -*- coding:utf-8 -*-
import os
import random
# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2 as cv
import numpy as np
from shutil import copyfile

lab_data=True


def calc_iou(bbox1, bbox2):
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)
    x_center1, y_center1, w1, h1 = np.split(bbox1, 4, axis=-1)
    xmin1 = (x_center1 - w1 / 2) * 640
    ymin1 = (y_center1 - h1 / 2) * 480
    xmax1 = (x_center1 + w1 / 2) * 640
    ymax1 = (y_center1 + h1 / 2) * 480

    x_center2, y_center2, w2, h2 = np.split(bbox2, 4, axis=-1)
    xmin2 = (x_center2 - w2 / 2) * 640
    ymin2 = (y_center2 - h2 / 2) * 480
    xmax2 = (x_center2 + w2 / 2) * 640
    ymax2 = (y_center2 + h2 / 2) * 480
    # xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
    # xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))

    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w

    union = area1 + np.squeeze(area2, axis=-1) - intersect
    return intersect / union


def delete_bbox(bbox1, bbox2, roi_bbox1, roi_bbox2, class1, class2, idx1, idx2, iou_value):
    idx = np.where(iou_value > 0.4)
    left_idx = idx[0]
    right_idx = idx[1]
    left = roi_bbox1[left_idx]
    right = roi_bbox2[right_idx]
    xmin1, ymin1, xmax1, ymax1, = np.split(left, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(right, 4, axis=-1)
    left_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    right_area = (xmax2 - xmin2) * (ymax2 - ymin2)
    left_idx = left_idx[np.squeeze(left_area < right_area, axis=-1)]  # 小的被删
    right_idx = right_idx[np.squeeze(left_area > right_area, axis=-1)]

    bbox1 = np.delete(bbox1, idx1[left_idx], 0)
    class1 = np.delete(class1, idx1[left_idx])
    bbox2 = np.delete(bbox2, idx2[right_idx], 0)
    class2 = np.delete(class2, idx2[right_idx])

    return bbox1, bbox2, class1, class2

def my_delete_box(bbox1, bbox2, cs, iou, obj_list):
    idx1 = np.arange(bbox1.shape[0])
    idx2 = np.arange(bbox2.shape[0])
    idx3 = np.arange(len(obj_list))
    idx = np.where(iou > 0)
    left_idx = idx[0]
    right_idx = idx[1]
    left = bbox1[left_idx]
    right = bbox2[right_idx]

    x_center1, y_center1, w1, h1 = np.split(left, 4, axis=-1)
    xmin1 = (x_center1 - w1 / 2) * 640
    ymin1 = (y_center1 - h1 / 2) * 480
    xmax1 = (x_center1 + w1 / 2) * 640
    ymax1 = (y_center1 + h1 / 2) * 480

    x_center2, y_center2, w2, h2 = np.split(right, 4, axis=-1)
    xmin2 = (x_center2 - w2 / 2) * 640
    ymin2 = (y_center2 - h2 / 2) * 480
    xmax2 = (x_center2 + w2 / 2) * 640
    ymax2 = (y_center2 + h2 / 2) * 480

    left_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    right_area = (xmax2 - xmin2) * (ymax2 - ymin2)

    left_idx = left_idx[np.squeeze(left_area < right_area, axis=-1)]  # 小的被删
    right_idx = right_idx[np.squeeze(left_area > right_area, axis=-1)]
    if right_idx.__contains__(bbox2.shape[0]-1):
        last_box_del = True
    else:
        last_box_del = False

    bbox1 = np.delete(bbox1, idx1[left_idx], 0)
    bbox2 = np.delete(bbox2, idx2[right_idx], 0)
    del_index = idx3[right_idx]
    print("delete idx2 and delete idx3")
    print(idx2[right_idx])
    print(idx3[right_idx])
    obj_list = [obj_list[i] for i in range(len(obj_list)) if i not in del_index]
    cs = np.delete(cs, idx2[right_idx], 0)
    return bbox2, cs, last_box_del, obj_list


def int_random(a, b, n) :
    # 定义一个空列表存储随机数
    a_list = []
    while len(a_list) < n :
        d_int = random.randint(a, b)
        if(d_int not in a_list) :
            a_list.append(d_int)
        else :
            pass
    return a_list

if __name__ == '__main__':
    if lab_data:
        bk_path = "/media/shenyl/Elements/sweeper/dataset/exp0624/rectified/images/"
    else:
        bk_path = "/media/shenyl/Elements/sweeper/dataset/0716/bk_croped/"
    obj_path = "/media/shenyl/Elements/sweeper/dataset/0716/objects/"
    classes = ["badminton/", "wire_crop/", "mahjong/", "dogshit_crop/", "carpet/"]
    result_path = "/media/shenyl/Elements/sweeper/dataset/0716/result/"
    result_img_path = result_path + "images/"
    result_label_path = result_path + "labels/"
    if not os.path.exists(result_img_path):
        os.makedirs(result_img_path)
    if not os.path.exists(result_label_path):
        os.makedirs(result_label_path)

    if lab_data:
        bk_num = 400
        num_each_bk = 6
        num_each_class = 2
        count = 0
    else:
        bk_num = 12
        num_each_bk = 200
        num_each_class = 2
        # count = 0
        count = 2400

    iou = 0

    for b in range(bk_num):
        for img_num in range(num_each_bk):
            if lab_data:
                bk = cv.imread(bk_path + "%06d.jpg" % b)
            else:
                bk = cv.imread(bk_path + "%02d.jpg" % b)
            # if bk.shape[2] == 3:
            #     bk = cv.cvtColor(bk,cv.COLOR_BGR2GRAY)
            bk_h, bk_w,_ = bk.shape
            boxes = np.array([])
            obj_list = []
            cs = np.arange(len(classes))
            cs = np.repeat(cs, num_each_class)
            obj_index = 0
            for i in range(len(classes)):
                c = classes[i]
                class_path = obj_path + c
                ObjectsList = os.listdir(class_path)
                # print(ObjectsList)
                ObjectNum = len(ObjectsList)
                # print(ObjectNum)
                idx_list = int_random(0, ObjectNum-1, num_each_class)
                for j in idx_list:
                    obj_file_path = class_path + "%06d.png"%j
                    # obj_file_path = "/media/shenyl/Elements/sweeper/dataset/0716/objects/new_dogshit_rename/000007.png"
                    print(obj_file_path)
                    obj = cv.imread(obj_file_path)
                    # if obj.shape[2] == 3:
                    #     obj = cv.cvtColor(obj, cv.COLOR_BGR2GRAY)
                    # cv.imshow("obj", obj)
                    # if cv.waitKey(0) == 27:
                    #     cv.destroyWindow("obj")
                    # todo: augumentation objects
                    # 缩放
                    h, w, _= obj.shape
                    scale = random.uniform(0.5, 1.5)
                    obj = cv.resize(obj,(int(w*scale),int(h*scale)))
                    # cv.imshow("obj after resize", obj)
                    # if cv.waitKey(0) == 27:
                    #     cv.destroyWindow("obj")
                    # 旋转


                    # 翻转
                    if not i == 4:
                        flip = random.uniform(0,3)
                        if i == 3:
                            flip = 1.5
                        if flip<1:
                            obj = cv.flip(obj, 1)  # 水平翻转
                        if flip>2:
                            obj = cv.flip(obj, 0)  # 垂直翻转
                    # cv.imshow("obj after flip", obj)
                    # if cv.waitKey(0) == 27:
                    #     cv.destroyWindow("obj")
                    # 变色调


                    # 将物体patches放置于背景图片上
                    # 小物体放在远处 area<1500, idy:[bk_h/2, 3*bk_h/4-h]
                    # 大物体放在近处 area>1500, idy:[3*bk_h/4, bk_h-h]
                    # print(obj.shape[0]*obj.shape[1])
                    h, w, _= obj.shape
                    area = h*w
                    # print("obj_index "+str(obj_index))
                    if area < 400:
                        cs = np.delete(cs, obj_index, 0)
                        continue
                    if area>1500:
                        idy = int_random(max(3*bk_h/4-h, 0), bk_h - h, 1)[0]
                    else:
                        idy = int_random(2*bk_h/4, 3*bk_h/4, 1)[0]
                    idx = int_random(0, bk_w-w, 1)[0]
                    box = np.array([[(idx+w/2)/bk_w, (idy+h/2)/bk_h, w/bk_w, h/bk_h]])
                    if boxes.shape[0] == 0:
                        obj_list = [obj]
                        boxes = box
                    else:
                        boxes = np.append(boxes, box, axis=0)
                        obj_list.append(obj)


                    obj_index = obj_index + 1


            iou = calc_iou(boxes, boxes)
            boxes, cs, last_box_del, obj_list = my_delete_box(boxes, boxes, cs, iou, obj_list)

            result = bk
            assert len(obj_list) == boxes.shape[0]
            for i, o in enumerate(obj_list):
                box = boxes[i]
                # w = int(box[2] * bk_w)
                # h = int(box[3] * bk_h)
                w = o.shape[1]
                h = o.shape[0]
                idx = int(box[0] * bk_w - w / 2)
                idy = int(box[1] * bk_h - h / 2)

                obj_grey = cv.cvtColor(o, cv.COLOR_BGR2GRAY)  #
                _, mask = cv.threshold(obj_grey, 1, 255, cv.THRESH_BINARY)
                mask_inv = cv.bitwise_not(mask)

                part_bk = bk[idy:idy + h, idx:idx + w, :]
                bk_result = np.ones(part_bk.shape)
                print("part_bk shape and mask_inv shape")
                print(part_bk.shape)
                print(mask_inv.shape)
                img_bg = cv.bitwise_and(part_bk, part_bk, mask=mask_inv)
                result[idy:idy + h, idx:idx + w, :] = img_bg + o


            # save result
            print("finish one frame")
            # cv.imshow("result", result)
            cv.imwrite(result_img_path + "%06d.jpg"%count, result)
            # if cv.waitKey(0) == 27:
            #     cv.destroyWindow("result")
            assert boxes.shape[0] == cs.shape[0]

            if lab_data:
                source_file = "/media/shenyl/Elements/sweeper/dataset/exp0624/rectified/labels/" + "%06d.txt"%b
                target_file = result_label_path+"%06d.txt"%count
                copyfile(source_file, target_file)
            if lab_data:
                with open(result_label_path + "%06d.txt" % count, 'a') as f:
                    for i in range(boxes.shape[0]):
                        f.write(str(cs[i]) + " " + str(boxes[i, 0]) + " " + str(boxes[i, 1]) + " " + str(
                            boxes[i, 2]) + " " + str(boxes[i, 3]))
                        f.write("\n")
            else:
                with open(result_label_path + "%06d.txt"%count, 'w') as f:
                    for i in range(boxes.shape[0]):
                        f.write(str(cs[i])+" "+str(boxes[i,0])+" "+str(boxes[i,1])+" "+str(boxes[i,2])+" "+str(boxes[i,3]))
                        f.write("\n")
            count = count+1

