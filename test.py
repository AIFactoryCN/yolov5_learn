import torch
from tqdm import tqdm
from models import heads
from models.model import Model
from dataloader import createDataLoader

def run(model, head, test_loader, device=None):
    
    
    for batch_index, (images, targets, paths, shapes) in enumerate(tqdm(test_loader)):

        images = images.to(torch.float32)
        targets = targets

        predicts = model(images)
        objs = head.detect(predicts)
        


    
    pass




if __name__ == "__main__":
    model = Model("yamls/yolov5s.yaml")
    model.eval()
    head = heads.Head()
    test_loader, dataSet = createDataLoader("/data/data_01/shituo/data/Mouse/mouse/test_list_learn.txt", 640, 2, False)
    run(model, head, test_loader)
    
    
