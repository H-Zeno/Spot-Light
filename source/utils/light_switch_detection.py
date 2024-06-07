import numpy as np
import cv2
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from source.utils.object_detetion import BBox

def _filter_detections_YOLOWorld(detections):

    # squaredness filter
    squaredness = (np.minimum(detections.xyxy[:,2] -detections.xyxy[:,0], detections.xyxy[:,3] -detections.xyxy[:,1])/
                   np.maximum(detections.xyxy[:,2] -detections.xyxy[:,0], detections.xyxy[:,3] -detections.xyxy[:,1]))

    idx_dismiss = np.where(squaredness < 0.95)[0]

    filtered_detections = sv.Detections.empty()
    filtered_detections.class_id = np.delete(detections.class_id, idx_dismiss)
    filtered_detections.confidence = np.delete(detections.confidence, idx_dismiss)
    filtered_detections.data['class_name'] = np.delete(detections.data['class_name'], idx_dismiss)
    filtered_detections.xyxy = np.delete(detections.xyxy, idx_dismiss, axis=0)

    return filtered_detections

def filter_detections_ultralytics(detections):

    detections = detections[0].cpu()
    xyxy = detections.boxes.xyxy.numpy()

    # filter squaredness outliers
    squaredness = (np.minimum(xyxy[:, 2] - xyxy[:, 0],
                              xyxy[:, 3] - xyxy[:, 1]) /
                   np.maximum(xyxy[:, 2] - xyxy[:, 0],
                              xyxy[:, 3] - xyxy[:, 1]))

    #filter area outliers
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

    # todo filter bounding boxes inside of larger bounding boxes

    idx_keep = np.where(np.logical_and(areas < 2*np.median(areas), squaredness > 0.88))[0]

    bbox = xyxy[idx_keep,:]

    return bbox



def predict_light_switches(image: np.ndarray, model_type: str = "yolov9c", vis_block: bool = False):

    if model_type == "yolo_world/l":
        classes = ["round light switch",
                   "white light switch",
                   "small light switch",
                   "push button light switch",
                   "rotating light switch",
                   "turning round  light switch",
                   "double push button light switch",
                   "black light switch"]

        model = YOLOWorld(model_id=model_type)
        model.set_classes(classes)

        def _callback(image_slice: np.ndarray) -> sv.Detections:
            result = model.infer(image_slice, confidence=0.005)
            return sv.Detections.from_inference(result).with_nms(threshold=0.05, class_agnostic=True)

        slicer = sv.InferenceSlicer(callback=_callback, slice_wh=(image.shape[0] // 1, image.shape[1] // 1),
                                    overlap_ratio_wh=(0.2, 0.2))

        detections = slicer(image)
        detections = _filter_detections_YOLOWorld(detections=detections)

        BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2, color=sv.Color.RED)
        LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=1, text_scale=1, text_color=sv.Color.BLACK)

        if vis_block:
            annotated_image = image.copy()
            annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
            # annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
            sv.plot_image(annotated_image, (20, 20))

        return detections

    elif model_type == "yolov9c":
        model = YOLO('/home/cvg-robotics/tim_ws/YOLO-World/runs/detect/train12/weights/best.pt')
        results_predict = model.predict(source=image, imgsz=1280, conf=0.3, iou=0.4, max_det=8, agnostic_nms=True,
                                        save=False)  # save plotted images

        boxes = filter_detections_ultralytics(detections=results_predict)

        if vis_block:
            canv = image.copy()
            for box in boxes:
                xB = int(box[2])
                xA = int(box[0])
                yB = int(box[3])
                yA = int(box[1])

                cv2.rectangle(canv, (xA, yA), (xB, yB), (0, 255, 0), 2)

            bbs = []
            for box in boxes:
                bbs.append(BBox(box[0], box[1], box[2], box[3]))

            plt.imshow(canv)
            plt.show()

        return bbs

if __name__ == "__main__":
    image = cv2.imread("/home/cvg-robotics/tim_ws/IMG_0519.jpeg")
    model_type = "yolov9c"
    detections = predict_light_switches(image, model_type, vis_block=True)
    a = 2