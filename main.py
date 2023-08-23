import cv2
import obd
import torch
import yaml
from gpiozero import DistanceSensor
from time import sleep
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov7.utils.plots import color_list, plot_one_box


# Checking if the engine is running before doing anything else
def is_engine_running(connection):
    cmd = obd.commands.RUN_TIME
    response = connection.query(cmd)

    if response.is_null():
        return False
    else:
        run_time = response.value.magnitude
        return run_time > 0


# Function to detect obstacles using the ultrasonic sensor
def detect_obstacles(sensor):
    distance = sensor.distance * 100
    if distance < 60:  # Adjust the distance threshold
        print("Obstacle detected at {:.2f} cm".format(distance))
        camera_activate()
    else:
        print("No obstacle detected")
    sleep(0.1)  # Adjust the delay as needed


with open('yolov7/data/coco.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    names = data['names']


# Camera activation function
def camera_activate():
    weights = 'yolov7/yolov7.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(weights, map_location=device)  # Load the YOLOv7 model
    model.to(device).eval()

    live_camera = cv2.VideoCapture(0)
    if not live_camera.isOpened():
        print("The camera sensors are not running!")
        return

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


# Main function to perform tasks when the engine is running
def perform_tasks_when_engine_running():
    # Initialize the OBD-II connection
    connection = obd.OBD()

    if is_engine_running(connection):
        print("Engine is running. Proceeding with tasks...")

        # Initialize the ultrasonic sensor on appropriate pins
        trigger_pin = 17       # Maybe changed later
        echo_pin = 27          # Maybe changed later
        ultrasonic_sensor = DistanceSensor(echo=echo_pin, trigger=trigger_pin)

        # Call the obstacle detection function within this block
        detect_obstacles(ultrasonic_sensor)

    else:
        print("Engine is not running.")


if __name__ == "__main__":
    perform_tasks_when_engine_running()


"""
Throughout this program, I used a method of making such a sequence of functions such that one 
will have to be called in case another one has just been executed. This kind of schematic structure
will just speed up the program, save processing time and resources by preventing some functions to run 
when not they are not currently needed!
"""