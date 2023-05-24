+ Cài đặt môi trường :

Chạy lệnh sau để cài đặt các thư viện cần thiết cho chương trình
```
pip install -r setup.txt
```
- file setup.txt là file chứa các thư viện và công cụ có sẵn cần thiết cho chương trình.

### Note : thư viện tensorflow có hai phiên bản là tensorflow và tensorflow-gpu, nếu máy tính có GPU thì cần cài đặt thêm bộ cuda toolkit và cuDNN của NVIDIA để có thể sử dụng tensorflow-gpu. Link download : https://developer.nvidia.com/cuda-downloads

## Do KNN là thuật toán tham lam nên nó sẽ lưu toàn bộ dữ liệu train vào bộ nhớ, do đó kích thước của model KNN lớn nên nhóm đã xóa mô hình KNN, hãy chạy `python KNN.py` để tạo lại model KNN trước khi test.

+ Chạy chương trình :

    - Chạy lệnh `python NN.py` trên terminal để train model và lưu model CNN và Mạng neural thông thường vào thư mục model.(Lưu ý khi fit hàm, có thể sử dụng callback để dừng train và kiểm tra thời gian mà 2 mô hình đạt acc = 99% bằng cách comment và uncomment dòng 73 và 74, dòng 88 và 90 trong file NN.py)
    - Chạy lệnh `python KNN.py` trên terminal để train model KNN và lưu model vào thư mục model.
    - Chạy lenh `python softmaxRegression.py` trên terminal để train model softmax regression và lưu model vào thư mục model.
    - Chạy lệnh `python test.py` trên terminal để sử dụng các model để dự đoán một số ảnh trong datatest và xem kết quả về các độ đo như accuray, recall, precision, f1-score  để đưa ra đánh giá.

    - Chạy lệnh `python Apply.py` trên terminal để sử dụng model để dự đoán một ảnh chữ số viết tay(các ảnh viết tay nằm trong 
      thư mục images) và xem kết quả.(Chú ý thay đổi đường dẫn ảnh trong file Apply.py để dự đoán các ảnh khác)

    - File "UI.py" là file chạy giao diện người dùng, chạy lệnh `python UI.py` trên terminal để chạy giao diện người dùng để có thể chọn các ảnh viết tay khác thay vì thay đổi đường dẫn. 

    - File `Convolution_Maxpooling.ipynb` mô phỏng cách Convolution và Maxpooling hoạt động.

    - Lưu ý: Thay đổi các epochs trong các file `NN.py, softmaxRegression.py` và thay đổi các k_neiborgh trong file `KNN` để kiểm tra hiệu quả của các mô hình với các bộ tham số khác nhau.
