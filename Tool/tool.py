from ultralytics import YOLO
import cv2
import os

# Tải mô hình YOLOv8 đã huấn luyện

file_path = r"...\Detection_Food\Model\best.pt"
# r"D:\Model\Detection_Food\yolov8m.pt"
model = YOLO(file_path)

# Mở video
# D:\Detection_Food\Video_test\1110484869-preview.mp4
video_path = r"...\Detection_Food\Video_test\1107836275-preview.mp4"
# r'D:\Detection_Food\Video_test\1110484869-preview.mp4'
cap = cv2.VideoCapture(video_path)

# Kiểm tra nếu video không thể mở
if not cap.isOpened():
    print("Error: Không thể mở video.")
    exit()

# Tạo thư mục để lưu file .txt và ảnh đã đánh nhãn nếu chưa tồn tại
annotated_images_dir = r'...\Detection_Food\Dataset\Apple\frames'
output_dir = r'...\Detection_Food\Dataset\Apple\labels'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(annotated_images_dir, exist_ok=True)

# Đọc video và dự đoán trên từng khung
frame_count = 1
while cap.isOpened():
    ret, frame = cap.read()  # Đọc từng khung hình của video
    if not ret:
        print("Không còn khung hình để đọc hoặc lỗi khi đọc.")
        break

    # Thực hiện dự đoán trên kEhung hình
    results = model.predict(source=frame, show=False, conf=0.25)

    # Mở file .txt cho từng khung hình
    with open(os.path.join(output_dir, f'frame_{frame_count}.txt'), 'w') as f:
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Lấy tọa độ bounding box
            confidences = result.boxes.conf.cpu().numpy()  # Lấy độ tin cậy
            class_ids = result.boxes.cls.cpu().numpy()  # Lấy nhãn (class ID)
            height, width, _ = frame.shape  # Kích thước khung hình
            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                # x1, y1, x2, y2 = boxcheck(y1,y2,class_ids)
                x1, y1, x2, y2 = box
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

    # # Lưu ảnh đã đánh nhãn
    annotated_frame = results[0].plot()  # Vẽ kết quả dự đoán lên khung hình
    cv2.imwrite(os.path.join(annotated_images_dir, f'frame_{frame_count}.jpg'), annotated_frame)


    frame_count += 1

    # Hiển thị khung hình đã được dự đoán
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
