import numpy as np
import time

class MAPTool:
    def __init__(self, groundtruth_annotations, detection_annotations, class_names):
        """ 指定检测结果和gt的标注信息，以及labelmap
        # Arguments
            detection_annotations(dict): {image_id: [[left, top, right, bottom, confidence, classes_index], [left, top, right, bottom, confidence, classes_index]]} 
            groundtruth_annotations(dict): {image_id: [[left, top, right, bottom, 0, classes_index], [left, top, right, bottom, 0, classes_index]]}
            class_names(list): ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        """
        self.detection_annotations = detection_annotations
        self.groundtruth_annotations = groundtruth_annotations
        self.class_names = class_names
        self.average_precision_array = np.zeros((len(class_names), ))
        self.map_array = np.zeros((3, ))

        tic = time.time()
        self.compute()
        toc = time.time()
        self.compute_time = toc - tic  # second
        #print(f"MAP Compute time: {compute_time:.2f} second")

    def class_ap(self, class_name_or_index):
        ''' 返回指定类别的ap
        # Arguments:
            class_name_or_index(int or str): 如果指定为int则是类别索引，否则为类别名称

        # return:
            np.array([ap@0.5, ap@0.75, ap@0.5:0.95])
        '''
        class_index = class_name_or_index
        if isinstance(class_name_or_index, str):
            class_index = self.class_names.index(class_name_or_index)
        return self.average_precision_array[class_index]

    @property
    def map(self):
        ''' 
        # return:
            np.array([map@0.5, map@0.75, map@0.5:0.95])
        '''
        return self.map_array

    def iou(self, a, b):
        aleft, atop, aright, abottom = [a[i] for i in range(4)]
        awidth = aright - aleft + 1
        aheight = abottom - atop + 1

        bleft, btop, bright, bbottom = [b[i] for i in range(4)]
        bwidth = bright - bleft + 1
        bheight = bbottom - btop + 1

        cleft = np.maximum(aleft, bleft)
        ctop = np.maximum(atop, btop)
        cright = np.minimum(aright, bright)
        cbottom = np.minimum(abottom, bbottom)
        cross_area = (cright - cleft + 1).clip(0) * (cbottom - ctop + 1).clip(0)
        union_area = awidth * aheight + bwidth * bheight - cross_area
        return cross_area / union_area

    # methods: 'continuous', 'interp101', 'interp11'
    def integrate_area_under_curve(self, precision, recall, method="interp101"):
        """ Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rbgirshick/py-faster-rcnn.
        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        if method == 'interp101':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            #ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate，梯度积分，yolov5这么写的，https://blog.csdn.net/weixin_44338705/article/details/89203791
            ap = np.mean(np.interp(x, mrec, mpre))  # integrate，直接取均值，COCO工具计算用这个，能够小数位完全一样
        elif method == 'interp11':
            x = np.linspace(0, 1, 11)  # 11-point interp (VOC2007)
            ap = np.mean(np.interp(x, mrec, mpre))  # integrate，直接取均值，论文上都这么做的
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes (VOC2012)
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
        return ap

    def compute_average_precision(self, matched_table, sum_groundtruth, threshold):
        num_dets = len(matched_table)
        if num_dets == 0:
            return 0

        true_positive = np.zeros((num_dets, ))
        image_id_index = 3
        groundtruth_seen_map = {item[image_id_index] : set() for item in matched_table}
        for index, (confidence, matched_iou, matched_groundtruth_index, image_id) in enumerate(matched_table):
            image_base_seen_map = groundtruth_seen_map[image_id]
            if matched_iou >= threshold:
                if matched_groundtruth_index not in image_base_seen_map:
                    true_positive[index] = 1
                    image_base_seen_map.add(matched_groundtruth_index)

        num_predicts = np.arange(1, len(true_positive) + 1)
        accumulate_true_positive = np.cumsum(true_positive)
        precision = accumulate_true_positive / num_predicts
        recall = accumulate_true_positive / sum_groundtruth
        average_precision = self.integrate_area_under_curve(precision, recall)
        return average_precision

    def compute(self):
        ''' 计算MAP
        # return:
            np.array([map@0.5, map@0.75, map@0.5:0.95])
        '''
        average_precision_array = []
        max_dets = 100
        for classes in range(len(self.class_names)):

            matched_table = []
            sum_groundtruth = 0

            for image_id in self.groundtruth_annotations:
                dets = self.detection_annotations[image_id]
                gts = self.groundtruth_annotations[image_id]
                select_detection = dets[dets[:, -1] == classes]
                select_groundtruth = gts[gts[:, -1] == classes]
                
                num_detection = len(select_detection)
                num_groundtruth = len(select_groundtruth)

                num_use_detection = min(num_detection, max_dets)
                sum_groundtruth += num_groundtruth

                if num_detection == 0:
                    continue
                
                if len(select_groundtruth) == 0:
                    for index_of_detection in range(num_use_detection):
                        confidence = select_detection[index_of_detection, 4]
                        matched_table.append([confidence, 0, 0, image_id])
                    continue

                sgt = select_groundtruth.T.reshape(-1, num_groundtruth, 1)
                sdt = select_detection.T.reshape(-1, 1, num_detection)

                # num_groundtruth x num_detection
                groundtruth_detection_iou = self.iou(sgt, sdt)
                for index_of_detection in range(num_use_detection):
                    confidence = select_detection[index_of_detection, 4]
                    matched_groundtruth_index = groundtruth_detection_iou[:, index_of_detection].argmax()
                    matched_iou = groundtruth_detection_iou[matched_groundtruth_index, index_of_detection]
                    matched_table.append([confidence, matched_iou, matched_groundtruth_index, image_id])

            # sorted by confidence
            matched_table = sorted(matched_table, key=lambda x: x[0], reverse=True)
            ap_05 = self.compute_average_precision(matched_table, sum_groundtruth, 0.5)
            ap_075 = self.compute_average_precision(matched_table, sum_groundtruth, 0.75)
            ap_05_095 = np.mean([self.compute_average_precision(matched_table, sum_groundtruth, t) for t in np.arange(0.5, 1, 0.05)])
            average_precision_array.append([ap_05, ap_075, ap_05_095])

        self.average_precision_array = average_precision_array
        self.map_array = np.mean(average_precision_array, axis=0)
        return self.map_array
