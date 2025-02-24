import asyncio
response_queue = asyncio.Queue()
change_worker = False
request_tracker = None
workers = []
audio_handler = None
previous_worker = None
handle_messages_task = None
start_streaming_task = None