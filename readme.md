### I. Cấu trúc thư mục:
#### 1. Thư mục apis:
- Chứa code liên quan API cho việc upload 1 video từ ứng dụng mobile lên server Flask xử lý.
#### 2. Thư mục dataset:
- Chứa dataset (train, test) của 5 động tác: Plank, Pushup, Squat, Lunge, Bicep Curl.
- Thư mục này thực chất được tổng hợp từ các file csv (train_clean.csv và test.csv) từ các thư mục training/plank_model, training/pushup_model, training/squat_model, training/lunge_model, training/bicep_curl_model.
#### 3. Thư mục desktop_app:
- Chứa code cho việc xây dựng ứng dụng desktop và được viết bằng Tkinter.

#### 4. Cấu trúc folder trong thư mục training:
- Thư mục này chứa 5 folder đại diện cho 5 động tác: Plank, Pushup, Squat, Lunge, Bicep Curl.
- Lấy ví dụ về thư mục plank_model:
    + Thư mục audios: chứa file âm thanh hướng dẫn khắc phục lỗi sai cho động tác Plank.
    + Thư mục best_model: chứa model tốt nhất sau khi train.
    + Thư mục hyper_parameter: chứa các siêu tham số tốt nhất sau khi train.
    + File extract-data.ipynb dùng để trích xuất keypoint quan trọng cho từng động tác và các ảnh đầu vào sẽ lấy từ các thư mục mà tool gán nhãn tạo ra
    + sklearn_model.ipynb dùng để train model sử dụng các model Machine Learning của sklearn.
    + detection.ipynb: chứa code xử lý phương pháp hình học cho việc nhận diện lỗi sai cụ thể và đưa ra gợi ý sửa lỗi. Đây là phiên bản cuối cùng trước khi convert sang code python để sử dụng trong ứng dụng desktop.
    + File testing.ipynb: thử nghiệm model đã train với tập test.csv

### II. Cách chạy ứng dụng:
#### 1. Cài môi trường và thư viện:
- Cài môi trường ảo venv: `python -m venv .venv`
- Thực hiện cài đặt thư viện:
```python
# Kích hoạt môi trường ảo
Với window: ".\.venv\Scripts\activate"
Với MacOS, linux: "source .venv/bin/activate"

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Chạy ứng dụng:
#### 2.1. Chạy tool gán nhãn: 
+ Mục đích của tool này là với input là 1 video đầu vào thì sau khi chạy xong các cell của notebook này thì sẽ phân ra các frame vào các folder đại diện cho từng nhãn của động tác đó, ví dụ với động tác Plank thì sau khi chạy xong sẽ tách được các frame từ video vào các folder Correct, Wrong (đại diện cho 2 nhãn của động tác Plank).

+ Bước 1: Mở terminal và di chuyển đến thư mục labeling_tool
+ Bước 2: Chạy các cell trong thư mục `keypoints_extraction_tool.ipynb` cho việc đánh nhãn cho tập train. 
Hoặc với file `keypoints_extraction_tool_for_testing.ipynb` cho việc đánh nhãn cho tập test.
    
#### 2.2. Chạy ứng dụng desktop:
    + Bước 1: Mở terminal và di chuyển đến thư mục desktop_app.
    + Bước 2: Chạy lệnh: `python main.py`

#### 2.3. Chạy API:
    + Bước 1: Mở terminal và di chuyển đến thư mục apis.
    + Bước 2: Chạy lệnh `python main.py`
    + Sau khi chạy xong ở bước 2, API sẽ cung cấp 1 endpoint:
    '/upload/<tên động tác>', ví dụ '/upload/plank'.

