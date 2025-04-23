from ultralytics import YOLO
import cv2

# Perbaikan path model
model = YOLO('runs/detect/train/weights/best.pt')  # path relatif dari lokasi script

# Buka webcam
cap = cv2.VideoCapture(0)

# Mapping ID kelas ke nilai uang
label_map = {
    0: 1000,
    1: 2000,
    2: 5000,
    3: 10000,
    4: 20000,
    5: 50000,
    6: 100000
}

print("Deteksi dimulai... Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.5, verbose=False)[0]

    total = 0
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = box.conf[0].item()
        value = label_map.get(cls_id, 0)
        total += value

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"Rp{value} ({conf:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(frame, f"Total: Rp{total}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Deteksi Uang Kertas", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
