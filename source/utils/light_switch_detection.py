import numpy as np
import cv2
import supervision as sv
from tqdm import tqdm
from inference.models.yolo_world.yolo_world import YOLOWorld
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import torch
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
from skimage.transform import resize
from typing import Dict

def _filter_detections(detections):

    squaredness = (np.minimum(detections.xyxy[:,2] -detections.xyxy[:,0], detections.xyxy[:,3] -detections.xyxy[:,1])/
                   np.maximum(detections.xyxy[:,2] -detections.xyxy[:,0], detections.xyxy[:,3] -detections.xyxy[:,1]))

    idx_dismiss = np.where(squaredness < 0.95)[0]

    filtered_detections = sv.Detections.empty()
    filtered_detections.class_id = np.delete(detections.class_id, idx_dismiss)
    filtered_detections.confidence = np.delete(detections.confidence, idx_dismiss)
    filtered_detections.data['class_name'] = np.delete(detections.data['class_name'], idx_dismiss)
    filtered_detections.xyxy = np.delete(detections.xyxy, idx_dismiss, axis=0)

    return filtered_detections




def predict_light_switches(image: np.ndarray, vis_block: bool = False):

    classes = ["round light switch",
               "white light switch",
               "small light switch",
               "push button light switch",
               "rotating light switch",
               "turning round  light switch",
               "double push button light switch",
               "black light switch"]

    model = YOLOWorld(model_id="yolo_world/l")
    model.set_classes(classes)

    def _callback(image_slice: np.ndarray) -> sv.Detections:
        result = model.infer(image_slice, confidence=0.005)
        return sv.Detections.from_inference(result).with_nms(threshold=0.05, class_agnostic=True)

    slicer = sv.InferenceSlicer(callback=_callback, slice_wh=(image.shape[0] // 1, image.shape[1] // 1),
                                overlap_ratio_wh=(0.2, 0.2))

    detections = slicer(image)
    detections = _filter_detections(detections=detections)

    BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2, color=sv.Color.RED)
    LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=1, text_scale=1, text_color=sv.Color.BLACK)

    if vis_block:
        annotated_image = image.copy()
        annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
        # annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
        sv.plot_image(annotated_image, (20, 20))

    # Bbox = detections.xyxy
    #
    # switch_bbox = Bbox[4, :]
    # x1, y1, x2, y2 = switch_bbox.astype(int)
    # x, y, w, h = x1, y1, x2 - x1, y2 - y1
    #
    # cropped_image = image[y:y + h, x:x + w]
    # sv.plot_image(cropped_image, (20, 20))

    return detections

if __name__ == "__main__":
    image = cv2.imread("/home/cvg-robotics/tim_ws/data/switches/img_closeup_4.png")
    detections = predict_light_switches(image, vis_block=True)
    print(detections)