from ultralytics import YOLO
import cv2
import os

# Tải mô hình YOLOv8 đã huấn luyện
file_path = "/home/phamvanhung/Project_Github/test_model/best.pt"
model = YOLO(file_path)


# Tạo thư mục để lưu file .txt và ảnh đã đánh nhãn nếu chưa tồn tại
annotated_images_dir = '/home/phamvanhung/Project_Github/test_model/frames'
output_dir = '/home/phamvanhung/Project_Github/test_model/labels'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(annotated_images_dir, exist_ok=True)

#Doc du lieu trong folder
file_path = "/home/phamvanhung/Project_Github/test_model/Thư mục mới"
list_arr = os.listdir()

for i in range(len()):
    # Thực hiện dự đoán trên khung hình
    frame = cv2.imread(file_path + "/" + i)
    results = model.predict(source=frame, show=False, conf=0.25)

    # Mở file .txt cho từng khung hình
    with open(os.path.join(output_dir, '{i}.txt'.format(i)), 'w') as f:
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
    cv2.imwrite(os.path.join(annotated_images_dir, '{i}.jpg'.format(i)), annotated_frame)

