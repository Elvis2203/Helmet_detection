import cv2
from ultralytics import YOLO


# Find working webcam
def find_working_camera(max_index=10):
    for i in range(max_index):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF]:
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    print(f"Using camera index {i} with backend {backend}")
                    return i, backend
    return None, None

cam_index, cam_backend = find_working_camera()
if cam_index is None:
    raise RuntimeError("No working camera found!")


# Load YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")  # small COCO model


# Start webcam
cap = cv2.VideoCapture(cam_index, cam_backend)
if not cap.isOpened():
    raise RuntimeError("Failed to open webcam")

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Simulate helmet detection
    for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        if int(cls) == 0:  # Person detected
            x1, y1, x2, y2 = map(int, box)
            # Draw a "helmet" box on top of the head
            helmet_height = int((y2 - y1) * 0.25)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y1 + helmet_height), (0, 0, 255), 2)
            cv2.putText(annotated_frame, "Helmet", (x1 + 5, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display frame
    cv2.imshow("Person + Helmet Detection (Simulated)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
