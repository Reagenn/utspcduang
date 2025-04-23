from ultralytics import YOLO
import cv2
import time

# Load model
model = YOLO('runs/detect/train8/weights/best.pt')

# Kelas sesuai data.yaml
label_map = {
    0: 1000,
    1: 10000,
    2: 100000,
    3: 2000,
    4: 20000,
    5: 5000,
    6: 50000
}

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
print("Deteksi dimulai... Tekan 'q' untuk keluar.")

# Buffer deteksi
detected_labels = []
detected_timeout = 0
BUFFER_FRAMES = 15

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    results = model.predict(source=frame, conf=0.3, verbose=False)[0]
    end_time = time.time()

    # Jika ada deteksi, perbarui buffer
    if results.boxes:
        detected_labels = []
        total = 0
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            value = label_map.get(cls_id, 0)
            total += value
            detected_labels.append((box.xyxy[0], value, conf))
        detected_timeout = BUFFER_FRAMES
    else:
        # Kurangi timeout jika tidak ada deteksi
        if detected_timeout > 0:
            detected_timeout -= 1

    # Hitung ulang total
    total = sum([v for (_, v, _) in detected_labels])

    # Gambar semua dari buffer
    for bbox, value, conf in detected_labels:
        x1, y1, x2, y2 = map(int, bbox)
        label = f"Rp{value:,} ({conf:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Tampilkan Total & FPS
    fps = 1 / (end_time - start_time + 1e-6)
    cv2.putText(frame, f"Total: Rp{total:,}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    # Tampilkan frame
    cv2.imshow("Deteksi Uang Kertas", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
