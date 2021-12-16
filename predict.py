from keras.layers import Input
from mask_rcnn import MASK_RCNN
from PIL import Image
from utils import utils
import datetime
import os
import numpy as np
from nets import mrcnn_training
import nets.mrcnn_training as modellib
from utils.config import Config
from nets.mrcnn_training import data_generator, load_image_gt
from dataset import ShapesDataset
from utils import visualize
import os.path as osp
import matplotlib.pyplot as plt

def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存txt文件成功")



class ShapesConfig(Config):
    NAME = "shapes"
    GPU_COUNT = 1
    # 应该通过设置IMAGES_PER_GPU来设置BATCH的大小，而不是下面的BATCH_SIZE
    # BATCHS_SIZE自动设置为IMAGES_PER_GPU*GPU_COUNT
    # 请各位注意哈！
    IMAGES_PER_GPU = 1
    # BATCH_SIZE = 1
    NUM_CLASSES = 1 + 2
    # RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512


if __name__ == "__main__":
    mask_rcnn = MASK_RCNN()
    config = ShapesConfig()
    datetime = datetime.datetime.now().strftime('%Y%m%d')

    # list = [2, 3, 4, 12, 13, 14, 15, 16, 17, 18, 19, 35, 36, 38, 58, 59, 63, 72, 91, 93, 96, 97, 98, 100, 103, 104, 105, 106, 107, 109, 110, 111, 132, 133, 134, 135, 136, 137, 153, 159, 175, 176, 178, 190, 192, 193, 194, 195, 196, 197, 200, 201, 202, 205, 206, 219, 221, 222, 226, 230, 231, 232, 233, 234, 235, 236, 243, 265, 267, 271, 272, 273, 274, 295, 296, 301, 302, 326, 330, 331, 332, 334, 337, 345, 355, 356, 360, 372, 373, 388, 389, 398, 399, 419, 420, 442, 450, 485, 486, 508, 509, 510, 512, 513, 517, 518, 519, 520, 521, 522, 529, 530, 532, 533, 536, 537, 538, 539, 540, 586, 596, 597, 606, 611, 620, 621, 652, 653, 654, 659, 666, 667, 668, 669, 673, 674, 690, 691, 701, 711, 718, 719, 726, 727, 728, 729, 732, 733, 734, 735, 755, 756, 761, 763, 764, 765, 766, 767, 769, 773, 781, 783, 784, 787, 788, 790, 794, 795, 800, 801, 802, 803, 804, 839, 855, 856, 857, 858, 859, 871, 872, 876, 877, 905, 906, 907, 911, 912, 937, 942, 943, 948]
    # list = [2, 3, 4, 12, 13, 14, 15, 16, 17, 18, 19, 35, 36, 38, 58, 59, 63, 72, 91, 96, 97, 103, 104, 105, 106, 107, 109, 110, 111, 132, 133, 134, 135, 136, 137, 153, 175, 176, 178, 190, 192, 193, 194, 195, 196, 197, 201, 202, 205, 206, 219, 221, 222, 226, 230, 231, 232, 233, 234, 235, 236, 243, 265, 267, 271, 272, 273, 274, 295, 296, 301, 302, 326, 330, 331, 332, 334, 337, 345, 355, 356, 360, 372, 373, 388, 389, 398, 399, 419, 420, 442, 450, 485, 486, 508, 509, 510, 512, 513, 517, 518, 519, 520, 521, 522, 529, 530, 533, 536, 537, 538, 539, 540, 586, 596, 597, 606, 611, 620, 621, 652, 653, 654, 659, 666, 667, 668, 669, 673, 674, 690, 691, 701, 711, 718, 719, 727, 728, 729, 732, 733, 734, 735, 755, 756, 761, 763, 764, 765, 766, 767, 769, 781, 783, 784, 787, 788, 790, 794, 795, 800, 801, 802, 803, 804, 839, 855, 856, 857, 858, 859, 871, 872, 876, 877, 905, 906, 907, 911, 912, 937, 942, 943, 948]
    # 出阴性结果
    # list = [5, 6, 7, 9, 10, 21, 23, 24, 26, 27, 30, 31, 32, 34, 43, 44, 46, 51, 52, 53, 55, 57, 66, 68, 69, 70, 74, 76, 77, 80, 82, 85, 87, 90, 92, 114, 120, 121, 122, 123, 125, 127, 128, 129, 130, 138, 142, 143, 145, 149, 152, 165, 168, 169, 171, 172, 179, 181, 191, 198, 199, 210, 211, 213, 215, 217, 223, 224,  227, 228, 229, 245, 247, 248, 253, 256, 257, 259, 263, 269, 270, 275, 276, 277, 280, 281, 282, 284, 285, 289, 294, 297, 303, 304, 309, 312, 313, 315, 316, 317, 319, 321, 324, 329, 333, 336, 340, 341, 344, 346, 347, 348, 359, 361, 366, 367, 368, 369, 370, 374, 375, 381, 387, 393, 397, 401, 404, 406, 408, 415, 416, 417, 422, 423, 424, 425, 426, 432, 439, 440, 444, 445, 446, 449, 451, 453, 459, 462, 466, 467, 469, 470, 473, 474, 477, 478, 480, 483, 487, 488, 490, 491, 492, 495, 497, 498, 499, 500, 505, 507, 511, 515, 516, 525, 527, 531, 534, 535, 542, 543, 546, 547, 548, 549, 550, 552, 555, 558, 559, 561, 562, 563, 565, 566, 567, 569, 572, 574, 575, 576, 577, 578, 588, 589, 592, 593, 595, 598, 607, 612, 614, 615, 626, 628, 629, 632, 634, 636, 637, 640, 642, 647, 655, 657, 658, 663, 672, 677, 679, 680, 681, 682, 683, 686, 689, 693, 698, 703, 706, 708, 720, 722, 741, 742, 743, 745, 747, 750, 751, 753, 754, 758, 775, 776, 778, 779, 782, 785, 786, 796, 797, 798, 806, 807, 811, 814, 818, 821, 822, 827, 830, 841, 842, 843, 845, 851, 852, 854, 860, 861, 862, 863, 864, 865, 866, 869, 886, 888, 893, 896, 897, 898, 899, 900, 908, 909, 915, 916, 917, 918, 922, 926, 934, 935, 939, 941, 946, 947, 950, 956, 957, 960]
    # list = [93,159,200,532,726,773,98,100]
    # list = [2, 3,4,5, 6, 7, 9, 10, 21, 23, 24, 26, 27, 30, 31, 32, 34, 43, 44, 46, 51, 52,]
    list=[2]
    print(len(list))
    data_list = []
    for id in list:
        data_list.append(str(id) + ".jpg")


    dataset_root_path = "./test_dataset_1015/"
    img_floder = dataset_root_path + "imgs/"
    mask_floder = dataset_root_path + "mask/"
    yaml_floder = dataset_root_path + "yaml/"
    #test_imglist = os.listdir(img_floder)
    imglist = os.listdir(img_floder)

    count = len(imglist)
    # np.random.seed(10101)
    # np.random.shuffle(imglist)
    test_imglist = imglist[:int(count * 1.0)]
    # test_imglist = data_list


    dataset_test = ShapesDataset()
    dataset_test.load_shapes(len(test_imglist), img_floder, mask_floder, test_imglist, yaml_floder)
    dataset_test.prepare()

    image_ids = dataset_test.image_ids
    APs = []
    Pcs = []
    Rcs =[]
    count1 = 0
    zero_count =0
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_test, config,
                                   image_id, use_mini_mask=False)
        if count1 == 0:
            save_box, save_class, save_mask = gt_bbox, gt_class_id, gt_mask
        else:
            save_box = np.concatenate((save_box, gt_bbox), axis=0)
            save_class = np.concatenate((save_class, gt_class_id), axis=0)
            save_mask = np.concatenate((save_mask, gt_mask), axis=2)

        molded_images = np.expand_dims(utils.mold_image(image, config), 0)
        # Run object detection

        # out_path = os.path.join("./predict_result", image)
        # image = mask_rcnn.detect_image(image)
        # image.save(out_path)
        # images = [molded_images]
        # image = [np.array(image)]
        results = mask_rcnn.detect_image(image)

        r = results

        # 将所有检测结果保存
        if count1 == 0:
            save_roi, save_id, save_score, save_m = r["rois"], r["class_ids"], r["scores"], r['masks']
        else:
            save_roi = np.concatenate((save_roi, r["rois"]), axis=0)
            save_id = np.concatenate((save_id, r["class_ids"]), axis=0)
            save_score = np.concatenate((save_score, r["scores"]), axis=0)
            save_m = np.concatenate((save_m, r['masks']), axis=2)
        print(count1)
        count1 += 1



        # 计算AP, precision, recall
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(save_box, save_class, save_mask,
                         save_roi, save_id, save_score, save_m,0.5)

    # f1_score = 2.0*(precisions*recalls)/(precisions+recalls)
    print("AP: ", AP)
    # print(" f1_score: ", f1_score)
    # APs.append(AP)



    # # 绘制PR曲线
    # plt.plot(recalls, precisions, 'b', label='PR')
    # plt.title('precision-recall curve')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend()
    # plt.show()
    #
    # # 保存precision, recall信息用于后续绘制图像
    # text_save('Kprecitest-0.85-0.5-old.txt', precisions)
    # text_save('Krecalltest-0.85-old.txt', recalls)
    # print("mAP: ", np.mean(AP))

    # for image_id in image_ids:
    #     image = dataset_test.load_image(image_id)
    #     mask, class_ids = dataset_test.load_mask(image_id)
    #     bbox = utils.extract_bboxes(mask)
    #     # for class_id in class_ids:
    #     #     ap = utils.compute_iou(gt_bbox, bbox, )
    #         # AP, precisions, recalls, overlaps = \
    #         #     utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
    #         #                            r["rois"], r["class_ids"], r["scores"], r['masks'])
    #
    #
    #     # Load image and ground truth data
    #     image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    #         modellib.load_image_gt(dataset_test, config,
    #                                image_id, use_mini_mask=False)
    #     molded_images = np.expand_dims(utils.mold_image(image, config), 0)
    #     # Run object detection
    #
    #     # out_path = os.path.join("./predict_result", image)
    #     # image = mask_rcnn.detect_image(image)
    #     # image.save(out_path)
    #     # images = [molded_images]
    #     # image = [np.array(image)]
    #     results = mask_rcnn.detect_image(image)
    #     r = results
    #     print("Results:", r)
    #     bboxs = utils.extract_bboxes(r['masks'])
    #
    #     drawed_image = visualize.display_instances(image[0],  bboxs, r['masks'], r['class_ids'],
    #                                               mask_rcnn.class_names, r['scores'])
    #
    # #     # Compute AP
    #     for result in r:
    #         AP, precisions, recalls, overlaps = \
    #             utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
    #                              bboxs, r["class_ids"], r["scores"], r['masks'])
    #         APs.append(AP)
    #         # Pcs.append(precisions)
    #         # recalls.append(recalls)
    #         # print("precisions :", precisions)
    #         # utils.compute_ap_range(gt_bbox)
    #
    #         if AP == 0.0:
    #             zero_count = zero_count + 1
    #         out_path = os.path.join("./predict_result",  str(count) + "_predict.jpg")
    #         #   image = mask_rcnn.detect_image(image)
    #         # drawed_image.save(out_path)
    #         print('Saved : %s' % str(count))
    #
    #         print("_______"+str(count)+"______")
    #         print("AP :", AP)
    #         count = count + 1
    #
    # print("mAP: ", np.array(APs).mean())
    # print("pre: ", np.array(Pcs).mean())
    # print("recall: ", np.array(Rcs).mean())
    # print("zero_AP_count: ", str(zero_count))

    # count = os.listdir("./predict/")
    # index = 0
    # date = datetime.datetime.now().strftime('%Y%m%d')
    # for i in range(0, len(count)):
    #     path = os.path.join("./predict", count[i])
    #     out_path = os.path.join("./predict_result", date + "_" + os.path.splitext(count[i])[0] + "_predict.jpg")
    #     if os.path.isfile(path) and path.endswith('jpg'):
    #         try:
    #             image = Image.open(path)
    #         except:
    #             print('Open Error! Try again!')
    #             continue
    #         else:
    #             index = index + 1
    #             image = mask_rcnn.detect_image(image)
    #             image.save(out_path)
    #             print('Saved : %s' % str(index))
    #
    # print('saved all')

    # img = input('Input image filename:')
    # try:
    #     image = Image.open(img)
    # except:
    #     print('Open Error! Try again!')
    #     continue
    # else:
    #     image = mask_rcnn.detect_image(image)
    #     image.show()
