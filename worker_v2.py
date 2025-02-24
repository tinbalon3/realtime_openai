import asyncio
from asyncio import Lock
import logging
from typing import List, Optional

class Worker:
    def __init__(self, worker, name):
        self.worker = worker
        self.name = name
        self.lock = Lock()  # Thêm khóa để đồng bộ hóa
        self.available_Status = True
        self.available_Used = False
        self.is_responsing = False

    async def set_status(self, available: bool, responding: bool, used: bool):
        async with self.lock:
            self.available_Status = available
            self.is_responsing = responding
            self.available_Used = used

async def get_available_worker(workers: List[Worker], current_worker: Worker, max_attempts: int = 5) -> Optional[Worker]:
    """Tìm worker khả dụng với cơ chế retry và ưu tiên worker chưa dùng."""
    attempt = 0
    while attempt < max_attempts:
        async with asyncio.Lock():  # Đảm bảo không có xung đột khi chọn worker
            # Ưu tiên worker chưa dùng, không đang xử lý, và khả dụng
            available_worker = next(
                (w for w in workers if w != current_worker and w.available_Status and not w.is_responsing and not w.available_Used),
                None
            )
            if available_worker:
                await available_worker.set_status(True, False, True)
                logging.info(f"Selected worker: {available_worker.name}")
                return available_worker

            # Nếu không có worker chưa dùng, chọn worker rảnh bất kỳ
            available_worker = next(
                (w for w in workers if w.available_Status and not w.is_responsing),
                None
            )
            if available_worker:
                await available_worker.set_status(True, False, True)
                logging.info(f"Reused worker: {available_worker.name}")
                return available_worker

        # Nếu không tìm thấy, chờ và thử lại
        logging.warning(f"No available worker on attempt {attempt + 1}/{max_attempts}")
        await asyncio.sleep(0.1)
        attempt += 1

    logging.error("No available worker after retries!")
    return None