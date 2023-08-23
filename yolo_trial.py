import cv2
import torch
import yaml
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov7.utils.plots import color_list, plot_one_box


with open('yolov7/data/coco.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    names = data['names']

weights = 'yolov7/yolov7.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights, map_location=device)  # Load the YOLOv7 model
model.to(device).eval()

live_camera = cv2.VideoCapture(0)
if not live_camera.isOpened():
    print("The camera sensors are not running!")

while True:
    ret, frame = live_camera.read()
    img = torch.from_numpy(frame).to(device)
    img = img.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.25, 0.45)

    # Process the YOLOv7 predictions and draw bounding boxes
    for det in pred[0]:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=color_list(int(cls)))
    cv2.imshow("Live Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

live_camera.release()
cv2.destroyAllWindows()



