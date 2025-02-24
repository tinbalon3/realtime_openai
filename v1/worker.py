from v1.realtime_client import RealtimeClient
class Worker:
    def __init__(self, worker: RealtimeClient, name: str):
        self.name = name
        self.worker = worker
        self.available_Status = True
        self.available_Used= False
        self.is_responsing = False