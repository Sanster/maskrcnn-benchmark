"""
My inference on coco with mask rcnn

good news is that, maskrcnn can run about 25 fps on GTX 1080ti
which is really very fast now!!

"""
import cv2
import torch
from torchvision import transforms as T
import os
import sys

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util
from maskrcnn_benchmark.config import cfg


# from dataset.coco import 
from alfred.vis.image.get_dataset_label_map import coco_label_map_list as CATEGORIES
from alfred.dl.torch.common import device
from alfred.vis.image.mask import draw_masks_maskrcnn
import time

class MaskRCNNDemo(object):

    def __init__(self, cfg, conf_thresh=0.7, min_size=224):
        self.cfg = cfg
        self.conf_thresh = conf_thresh
        self.min_size = 224
        self.cpu_device = torch.device("cpu")
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.model.to(device)
        
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

    def build_transform(self):
            cfg = self.cfg
            # we are loading images with OpenCV, so we don't need to convert them
            # to BGR, they are already! So all we need to do is to normalize
            # by 255 if we want to convert to BGR255 format, or flip the channels
            # if we want it to be in RGB in [0-1] range.
            if cfg.INPUT.TO_BGR255:
                to_bgr_transform = T.Lambda(lambda x: x * 255)
            else:
                to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])
            normalize_transform = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            transform = T.Compose(
                [
                    T.ToPILImage(),
                    T.Resize(self.min_size),
                    T.ToTensor(),
                    to_bgr_transform,
                    normalize_transform,
                ]
            )
            return transform
    
    def run_on_cv_img(self, image, show=False):
        if show:
            tic = time.time()
        in_image = self.transforms(image)
        image_list = to_image_list(in_image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(device)
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]
        # if show:
        #     print('fps: {}'.format(1/(time.time() - tic)))
        prediction = predictions[0]
        height, width = image.shape[:-1]
        prediction = prediction.resize((width, height))
        print('fps: {}'.format(1/(time.time() - tic)))
        
        boxes = prediction.bbox.numpy()
        scores = prediction.get_field('scores').numpy()
        labels = prediction.get_field('labels').numpy()
        masks = prediction.get_field('mask').squeeze(1).numpy()
        if show:
            res = draw_masks_maskrcnn(image, boxes, scores, labels, masks, human_label_list=CATEGORIES)
            # print('fps: {}'.format(1/(time.time() - tic)))
            return boxes, scores, labels, masks, res
        else:
            return boxes, scores, labels, masks, _

    
if __name__ == "__main__":
    cfg.merge_from_file('../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml')
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    cfg.freeze()
    mask_rcnn_demo = MaskRCNNDemo(cfg)
    
    v_f = sys.argv[1]
    cap = cv2.VideoCapture(v_f)

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_writer = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    while True:
        start_time = time.time()
        ret_val, img = cap.read()
        if ret_val:
            _, _, _, _, res = mask_rcnn_demo.run_on_cv_img(img, True)
            cv2.imshow("mask rcnn live", res)
            video_writer.write(res)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        else:
            print('Done!')
            exit(0)
    cv2.destroyAllWindows()
