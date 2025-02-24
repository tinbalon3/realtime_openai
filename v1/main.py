import asyncio
import os
import pyaudio
import wave
import json
import base64
import io
import queue
import threading
from typing import Optional, Callable, List, Dict, Any
from dotenv import load_dotenv
from enum import Enum
from pydub import AudioSegment
import websockets
from pynput import keyboard

from input_handler import InputHandler
from v2.request_tracker import RequestTracker
from v1.worker import Worker

load_dotenv()
import logging

# Configure logging
logging.basicConfig(filename='worker_status.log',encoding="utf-8", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
api_key = os.getenv("OPENAI_API_KEY")
response_queue = asyncio.Queue()

change_worker = False
request_tracker = RequestTracker()
workers = []
audio_handler = None
previous_worker = None
handle_messages_task = None
start_streaming_task = None
class TurnDetectionMode(Enum):
    SERVER_VAD = "server_vad"
    MANUAL = "manual"
class RealtimeClient:
    """Handles WebSocket communication with OpenAI Realtime API."""
    
    def __init__(
        self, 
        api_key: str,
        model: str = "gpt-4o-realtime-preview-2024-10-01",
        voice: str = "alloy",
       
        instructions: str = """You are a real-time translation tool. 
        Your only task is to translate text from Vietnamese to English accurately and concisely. 
        If the input is clear Vietnamese text, provide the English translation and keep any English words unchanged. 
        If the input contains noise, is unclear, or cannot be understood, return an empty string (''). 
        Do not provide explanations, extra comments, or anything beyond the translation or an empty string.""",
        temperature: float = 0.8,
        turn_detection_mode: TurnDetectionMode = TurnDetectionMode.MANUAL,
        on_text_delta: Optional[Callable[[str], None]] = None,
        on_audio_delta: Optional[Callable[[bytes], None]] = None,
        on_interrupt: Optional[Callable[[], None]] = None,
        on_input_transcript: Optional[Callable[[str], None]] = None,  
        on_output_transcript: Optional[Callable[[str], None]] = None,  
        extra_event_handlers: Optional[Dict[str, Callable[[Dict[str, Any]], None]]] = None
    ):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.ws = None
        self.on_text_delta = on_text_delta
        self.on_audio_delta = on_audio_delta
        self.on_interrupt = on_interrupt
        self.on_input_transcript = on_input_transcript
        self.on_output_transcript = on_output_transcript
        self.instructions = instructions
        self.temperature = temperature
        self.base_url = "wss://api.openai.com/v1/realtime"
        self.extra_event_handlers = extra_event_handlers or {}
        self.turn_detection_mode = turn_detection_mode
        self.tools  = []

        # Track current response state
        self._current_response_id = None
        self._current_item_id = None
        self._is_responding = False
        # Track printing state for input and output transcripts
        self._print_input_transcript = False
        self._output_transcript_buffer = ""
        

    async def connect(self) -> None:
        print("Connecting to Realtime API...")
        """Establish WebSocket connection with the Realtime API."""
        url = f"{self.base_url}?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        self.ws = await websockets.connect(url, additional_headers=headers)
        
        # Set up default session configuration
        tools = [t.metadata.to_openai_tool()['function'] for t in self.tools]
        for t in tools:
            t['type'] = 'function'  # TODO: OpenAI docs didn't say this was needed, but it was

        
        if self.turn_detection_mode == TurnDetectionMode.MANUAL:
            await self.update_session({
                "modalities": ["text", "audio"],
                "instructions": self.instructions,
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "tools": tools,
                "tool_choice": "auto",
                "temperature": self.temperature,
            })
        elif self.turn_detection_mode == TurnDetectionMode.SERVER_VAD:
            
            await self.update_session({
                "modalities": ["text", "audio"],
                "instructions": self.instructions,
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 200,
                    "silence_duration_ms": 300
                },
                "tools": tools,
                "tool_choice": "auto",
                "temperature": self.temperature,
            })
        else:
            raise ValueError(f"Invalid turn detection mode: {self.turn_detection_mode}")

    async def update_session(self, config: Dict[str, Any]) -> None:
        
        """Update session configuration."""
        event = {
            "type": "session.update",
            "session": config
        }
        await self.ws.send(json.dumps(event))

    async def send_audio(self, audio_chunk: bytes):
        """Send streaming audio to the API."""
        audio_b64 = base64.b64encode(audio_chunk).decode()
        await self.ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))

    async def handle_messages(self):
        """Handle messages from OpenAI WebSocket."""
        print(f"Handling messages...")
        global audio_handler,request_tracker
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type")

                if event_type == "conversation.item.created":
                  
                    self._current_item_id = event.get("item", {}).get("id")
                    self._is_responding = True
                    request_id = request_tracker.new_request()  # Tạo ID ngay khi gửi yêu cầu
                    await request_tracker.track_request(request_id, self._current_item_id)  # Liên kết ID với response_id
                    # print(f"Created item {self._current_item_id} for request {request_id} of worker {worker.name}")
                    worker = next((w for w in workers if w.worker == self), None)
                    if worker:
                        worker.available_Status = False  # Mark as unavailable
                        worker.is_responsing = True
                        

                
                elif event_type == "input_audio_buffer.speech_started":
                    print("Speech started\n")
                    
                elif event_type == "input_audio_buffer.speech_stopped":
                    print("Speech ended\n")
                    
                elif event_type == "response.done":
                    worker = next((w for w in workers if w.worker == self), None)
                    if worker:
                        worker.is_responsing = False
                        worker.available_Status = True  
                        
                elif event_type == "response.audio_transcript.done":
                    transcript = event.get("transcript", "")
                    response_id = event.get("item_id", "")
                    request_id = request_tracker.get_request_id(response_id)
                    # print(f"Received response_id {response_id} for request_id {request_id}")
                    self._is_responding = False
                    worker = next((w for w in workers if w.worker == self), None)
                    if worker:
                        worker.is_responsing = False
                        worker.available_Status = True
                    if request_id is not None:
                        await request_tracker.add_response(request_id, transcript)  # Chỉ xử lý nếu transcript có nội dung

        except websockets.exceptions.ConnectionClosed:
            print(f"Connection closed ")
        except Exception as e:
            print(f"Error in message handling : {str(e)}")


    async def close(self):
        """Close WebSocket connection."""
        if self.ws:
            await self.ws.close()
    


class AudioHandler:
    """Handles audio input/output for streaming and playback."""
    def __init__(self):
        # Audio parameters
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 24000
        self.chunk = 24000

        self.audio = pyaudio.PyAudio()

        # Recording params
        self.recording_stream: Optional[pyaudio.Stream] = None
        self.recording_thread = None
        self.recording = False

        # streaming params
        self.streaming = False
        self.stream = None

        # Playback params
        self.playback_stream = None
        self.playback_buffer = queue.Queue(maxsize=50)
        self.playback_event = threading.Event()
        self.playback_thread = None
        self.stop_playback = False
       
    
    async def start_streaming(self, worker: Worker):
        """Start continuous audio streaming with efficient worker switching."""
        if self.streaming:
            return

        self.streaming = True
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        print(f"\nStreaming audio... Press 'q' to stop.")

        global workers
        
        while self.streaming:
            try:
                # Kiểm tra nếu worker hiện tại không khả dụng hoặc đang phản hồi
                if not worker.available_Status and worker.available_Used:
                    new_worker = await get_available_worker(workers,worker)

                    if new_worker:
                        # print(f"Chuyển từ {worker.name} sang {new_worker.name}")
                        worker.available_Used = False
                        worker = new_worker
                        worker.available_Used = True
                    else:
                        print("Không có worker khả dụng")
                        break

                # Đọc dữ liệu âm thanh
                data = self.stream.read(self.chunk, exception_on_overflow=False)

                # Gửi dữ liệu đến worker
                # print(f"Sending audio to {worker.name} and worker status: {worker.available_Status} and worker used: {worker.available_Used}")
                
                await worker.worker.send_audio(data)

            except Exception as e:
                print(f"Lỗi khi streaming: {e}")
                break

            await asyncio.sleep(0.01)  # Tránh chiếm toàn bộ CPU
 
        
            
    def play_audio(self, audio_data: bytes):
        """Add audio data to the buffer"""
        try:
            self.playback_buffer.put_nowait(audio_data)
        except queue.Full:
            # If the buffer is full, remove the oldest chunk and add the new one
            self.playback_buffer.get_nowait()
            self.playback_buffer.put_nowait(audio_data)
        
        if not self.playback_thread or not self.playback_thread.is_alive():
            self.stop_playback = False
            self.playback_event.clear()
            self.playback_thread = threading.Thread(target=self._continuous_playback)
            self.playback_thread.start()
            
    def _continuous_playback(self):
        """Continuously play audio from the buffer"""
        self.playback_stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            output=True,
            frames_per_buffer=self.chunk
        )

        while not self.stop_playback:
            try:
                audio_chunk = self.playback_buffer.get(timeout=0.1)
                self._play_audio_chunk(audio_chunk)
            except queue.Empty:
                continue
            
            if self.playback_event.is_set():
                break

        if self.playback_stream:
            self.playback_stream.stop_stream()
            self.playback_stream.close()
            self.playback_stream = None
    def _play_audio_chunk(self, audio_chunk):
        try:
            # Convert the audio chunk to the correct format
            audio_segment = AudioSegment(
                audio_chunk,
                sample_width=2,
                frame_rate=24000,
                channels=1
            )
            
            # Ensure the audio is in the correct format for playback
            audio_data = audio_segment.raw_data
            
            # Play the audio chunk in smaller portions to allow for quicker interruption
            chunk_size = 1024  # Adjust this value as needed
            for i in range(0, len(audio_data), chunk_size):
                if self.playback_event.is_set():
                    break
                chunk = audio_data[i:i+chunk_size]
                self.playback_stream.write(chunk)
        except Exception as e:
            print(f"Error playing audio chunk: {e}")
            
    def stop_streaming(self):
        """Stop audio streaming."""
        self.streaming = False
        
    def stop_playback_immediately(self):
        """Stop audio playback immediately."""
        self.stop_playback = True
        self.playback_buffer.queue.clear()  # Clear any pending audio
        self.currently_playing = False
        self.playback_event.set()
    def cleanup(self):
        """Clean up audio resources"""
        self.stop_playback_immediately()

        self.stop_playback = True
        if self.playback_thread:
            self.playback_thread.join()

        self.recording = False
        if self.recording_stream:
            self.recording_stream.stop_stream()
            self.recording_stream.close()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        self.audio.terminate()
        
    def stop_streaming(self):
        """Stop audio streaming."""
        self.streaming = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        



async def init_client():
    """Initialize and manage workers and streaming."""
    global workers,audio_handler,request_tracker
    print("Initializing clients...")
    audio_handler = AudioHandler()
    input_handler = InputHandler()
    input_handler.loop = asyncio.get_running_loop()
    asyncio.create_task(process_responses())
    client_1 = Worker(RealtimeClient(
        api_key=api_key,
        on_audio_delta=lambda audio: audio_handler.play_audio(audio),
        on_interrupt=lambda: audio_handler.stop_playback_immediately(),
        turn_detection_mode=TurnDetectionMode.SERVER_VAD
        
    ),"worker_1")

    client_2 = Worker(RealtimeClient(
        api_key=api_key,
        on_audio_delta=lambda audio: audio_handler.play_audio(audio),
        on_interrupt=lambda: audio_handler.stop_playback_immediately,
        turn_detection_mode=TurnDetectionMode.SERVER_VAD  
    ),"worker_2")
    client_3 = Worker(RealtimeClient(
        api_key=api_key,
        on_audio_delta=lambda audio: audio_handler.play_audio(audio),
        on_interrupt=lambda: audio_handler.stop_playback_immediately,
        turn_detection_mode=TurnDetectionMode.SERVER_VAD  
    ),"worker_3")
   
    workers = [client_1,client_2,client_3]
    

    worker_used = workers[0]
    worker_used.available_Used = True
    listener = keyboard.Listener(on_press=input_handler.on_press)
    listener.start()
    try:
        print("Connected to OpenAI Realtime API!")
        print("Press 'q' to quit\n")
        for worker in workers:
            await worker.worker.connect()
            asyncio.create_task(worker.worker.handle_messages())
       
        asyncio.create_task(audio_handler.start_streaming(worker_used))
        
        
        while True:
            command, _ = await input_handler.command_queue.get()
            
            if command == 'q':
                break
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        audio_handler.stop_streaming()
        audio_handler.cleanup()
        await close_connect()
        
async def get_available_worker(workers,worker_used):
    """Find the first available worker or reset usage if all are used."""
    # First try to find an unused worker
    available_worker = next(
        (worker for worker in workers if worker != worker_used and worker.available_Status and not worker.is_responsing and not worker.available_Used),
        None
    )
    
    if not available_worker:
        idle_workers = [worker for worker in workers if worker.available_Status and not worker.is_responsing]
        if idle_workers:
            # Reset the usage flag for all idle workers
            for worker in idle_workers:
                worker.available_Used = False
            # Get the first now-available worker
            available_worker = idle_workers[0]
    if available_worker is None:
        await log_worker_status(workers)
    return available_worker

async def process_responses():
    """Luôn lấy phản hồi từ request_tracker và xử lý theo thứ tự."""
    while True:
        request_id, response = await request_tracker.get_next_response()
        print(f"Processing response {request_id}: {response}")
        logging.info(f"Processing response {request_id}: {response}")

async def log_worker_status(workers):
    for worker in workers:
        logging.info(f"Worker {worker.name}:")
        logging.info(f"  Available Status: {worker.available_Status}")
        logging.info(f"  Available Used: {worker.available_Used}")
        logging.info(f"  Is Responding: {worker.is_responsing}")
        
async def close_connect():
    for worker in workers:
        await worker.worker.close()
        
if __name__ == "__main__":
    asyncio.run(init_client())
