# Real-time Translation System with OpenAI

Hệ thống dịch thuật realtime sử dụng OpenAI API để dịch song ngữ Anh-Việt.

## Tính năng

- Dịch realtime từ tiếng Anh sang tiếng Việt và ngược lại
- Hỗ trợ nhập liệu bằng giọng nói (Speech-to-Text)
- Phát hiện giọng nói tự động (Voice Activity Detection - VAD)
- Xử lý đa luồng với nhiều worker
- Hệ thống theo dõi và ghi log
- Tương thích với cả text và audio

## Yêu cầu hệ thống

- Python 3.8+
- OpenAI API key
- Microphone và loa
- Các thư viện Python:
  ```
  websockets
  pyaudio
  pydub
  python-dotenv
  pynput
  ```

## Cài đặt

1. Clone repository:
```bash
git clone <repository-url>
cd CloneRealTime
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

3. Tạo file .env và thêm API key:
```
OPENAI_API_KEY=your-api-key-here
```

## Sử dụng

1. Chạy phiên bản đơn worker:
```bash
python realtime_one_worker.py
```

2. Chạy phiên bản multi worker:
```bash
python realtime_mutil_worker.py
```

### Điều khiển
- Nhấn 'q' để thoát chương trình
- Nói vào microphone để bắt đầu dịch
- Chờ khoảng 1-2 giây sau khi nói để nhận kết quả

## Cấu hình

### Cài đặt Audio
```python
format = pyaudio.paInt16
channels = 1
rate = 24000
chunk = 4096
```

### Voice Activity Detection
```python
turn_detection = {
    "type": "server_vad",
    "threshold": 0.7,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 200
}
```

## Cấu trúc Project

- `realtime_one_worker.py`: Phiên bản đơn worker
- `realtime_mutil_worker.py`: Phiên bản đa worker
- `worker_v2.py`: Quản lý worker và trạng thái
- `input_handler.py`: Xử lý input từ bàn phím
- `request_tracker.py`: Theo dõi và quản lý request
- `global.py`: Các biến và hằng số toàn cục

## Xử lý Lỗi

- Kiểm tra microphone và loa được kết nối đúng
- Đảm bảo API key hợp lệ và còn quota
- Xem file `worker_status.log` để debug
- Kiểm tra kết nối internet

## Lưu ý

- Chất lượng dịch phụ thuộc vào chất lượng audio input
- Nên nói rõ ràng và hoàn chỉnh câu
- Tránh nhiễu và tiếng ồn xung quanh
- Đảm bảo internet ổn định

## License

MIT License