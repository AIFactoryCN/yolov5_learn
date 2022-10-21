import enum
import torch
from tqdm import tqdm
from models import heads
from models.model import Model
from dataloader import createDataLoader
from pathlib import Path
from util import *
import maptool

def run(model, head, test_loader, device=None):
    
    
    for batch_index, (images, targets, paths, shapes) in enumerate(tqdm(test_loader)):

        device_info = next(model.parameters()).device
        images = images.to(torch.float32)
        images = images.to(device_info)

        # targets.shape [image_index, class_index, cx, cy, width, height]
        targets = targets

        predicts = model(images)
        objs = head.detect(predicts)

        batch_size, _, height, width = images.shape  # batch size, channels, height, width
        targets[:, 2:] *= torch.Tensor([width, height, width, height])  # to pixels

        iouv = torch.linspace(0.5, 0.95, 10) # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        seen = 0
        stats = []
        groundtruth_annotations = {}
        detection_annotations = {}
        for image_index, pred in enumerate(objs):
            labels = targets[targets[:, 0] == image_index, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[image_index]), shapes[image_index][0]

            predn = pred.clone()
            scale_coords(images[image_index].shape[1:], predn[:, :4], shape, shapes[image_index][1])  # native-space pred
            
            predn[:, 0].clamp_(0, width)
            predn[:, 1].clamp_(0, height)
            predn[:, 2].clamp_(0, width)
            predn[:, 3].clamp_(0, height)
            image_id = image_index + batch_index * batch_size
            detection_annotations[image_id] = predn.cpu().numpy()

            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(images[image_index].shape[1:], tbox, shape, shapes[image_index][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
    
            for class_id, left, top, right, bottom, in labelsn:
                image_id = int(image_index) + batch_index * batch_size
                class_id = int(class_id)
                if image_id not in groundtruth_annotations:
                    groundtruth_annotations[image_id] = []

                groundtruth_annotations[image_id].append([left, top, right, bottom, 0, class_id])
        
        # merge groundtruth_annotations
        for image_id in groundtruth_annotations:
            groundtruth_annotations[image_id] = np.array(groundtruth_annotations[image_id], dtype=np.float32)

        label_map = ["mouse"]
        map_result = maptool.MAPTool(groundtruth_annotations, detection_annotations, label_map)
        map05, map075, map05095 = map_result.map
        print(f"map05: {map05:.3f}, map075: {map075:.3f}, map05095: {map05095:.3f}")
        model_score = map05 * 0.1 + map05095 * 0.9
        
    return model_score

    




if __name__ == "__main__":
    model = Model("yamls/yolov5s.yaml")
    model.eval()
    head = heads.Head()
    test_loader, dataSet = createDataLoader("/data/data_01/shituo/data/Mouse/mouse/test_list_learn.txt", 640, 2, 32, False)
    model_score = run(model, head, test_loader)
    print(model_score)
    
