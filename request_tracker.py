import asyncio


class RequestTracker:
    def __init__(self):
        self.request_counter = 0
        self.response_queue = asyncio.Queue()  # Hàng đợi ưu tiên để giữ thứ tự
        self.request_map = {}  # Lưu mapping từ request_id -> response_id

    def new_request(self):
        """Tạo request_id mới."""
        self.request_counter += 1
        return self.request_counter

    async def track_request(self, request_id, response_id):
        """Liên kết request_id với response_id."""
        self.request_map[response_id] = request_id

    def get_request_id(self, response_id):
        """Lấy request_id tương ứng với response_id."""
        return self.request_map.get(response_id, None)

    async def add_response(self, request_id, response):
        """Thêm phản hồi vào hàng đợi theo request_id."""
        await self.response_queue.put((request_id, response))

    async def get_next_response(self):
        """Lấy response tiếp theo theo thứ tự request_id."""
        return await self.response_queue.get()
