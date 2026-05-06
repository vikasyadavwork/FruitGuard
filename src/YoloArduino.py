import cv2
import serial
import time
import torch

from ultralytics import YOLO


# -------------------------
# CONFIG
# -------------------------

COM_PORT = "COM5"
BAUD_RATE = 9600

SAFE_DISTANCE_CM = 50.0
CONF_THRESHOLD = 0.50


# -------------------------
# CUDA / YOLO
# -------------------------

device = 0 if torch.cuda.is_available() else "cpu"

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = YOLO("yolo11n.pt")


# -------------------------
# ARDUINO
# -------------------------

arduino = serial.Serial(
    COM_PORT,
    BAUD_RATE,
    timeout=0.05
)

time.sleep(2)

distance = 999.0
last_command = ""


def send_command(command):
    global last_command

    if command != last_command:
        arduino.write((command + "\n").encode())
        print("Arduino:", command)
        last_command = command


# -------------------------
# CAMERA
# -------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera failed")
    exit()


while True:

    # -------------------------
    # READ ARDUINO DISTANCE
    # -------------------------

    while arduino.in_waiting > 0:

        line = arduino.readline().decode(
            errors="ignore"
        ).strip()

        if line.startswith("DIST:"):
            try:
                distance = float(line.split(":")[1])
            except ValueError:
                pass


    # -------------------------
    # READ CAMERA FRAME
    # -------------------------

    ok, frame = camera.read()

    if not ok:
        break


    # -------------------------
    # YOLO ON GPU
    # -------------------------

    results = model(
        frame,
        device=device,
        verbose=False
    )


    person_detected = False
    best_confidence = 0.0


    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # COCO class 0 = person
            if class_id == 0 and confidence >= CONF_THRESHOLD:

                person_detected = True

                best_confidence = max(
                    best_confidence,
                    confidence
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Person {confidence:.2f}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )


    # -------------------------
    # SENSOR FUSION
    # -------------------------

    if not person_detected:

        status = "SAFE"
        command = "GREEN"

    elif distance < SAFE_DISTANCE_CM:

        status = "DANGER"
        command = "RED"

    else:

        status = "PERSON DETECTED"
        command = "YELLOW"


    send_command(command)


    # -------------------------
    # UI
    # -------------------------

    cv2.putText(
        frame,
        f"Distance: {distance:.1f} cm",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Person confidence: {best_confidence:.2f}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "GPU Vision + Arduino",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
arduino.close()
cv2.destroyAllWindows()