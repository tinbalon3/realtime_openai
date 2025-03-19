import asyncio
import base64
import json
import os
import pyaudio
from request_tracker import RequestTracker
import queue
import threading
from typing import Optional, Callable, Dict, Any
from dotenv import load_dotenv
from enum import Enum
from pydub import AudioSegment
import websockets
from pynput import keyboard
from input_handler import InputHandler

load_dotenv()
import logging

# Configure logging
logging.basicConfig(filename='worker_status.log', encoding="utf-8", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
api_key = os.getenv("OPENAI_API_KEY")
print(type(api_key))
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
        instructions: str = """
   You are a real-time translation tool. Your only task is to translate text accurately and concisely between English and Vietnamese.

- If the input is in English, translate it to Vietnamese.
- If the input is in Vietnamese, translate it to English.
- If the input is meaningless, do not translate it and return an empty string ('').

Do not provide explanations, extra comments, or anything beyond the translation or an empty string.
        """,
        temperature: float = 0.8,
        turn_detection_mode: TurnDetectionMode = TurnDetectionMode.SERVER_VAD,
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
        self.tools = []
        self._current_response_id = None
        self._current_item_id = None
        self._is_responding = False
        self._print_input_transcript = False
        self._output_transcript_buffer = ""

    async def connect(self) -> None:
        print("Connecting to Realtime API...")
        url = f"{self.base_url}?model={self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}", "OpenAI-Beta": "realtime=v1"}
        self.ws = await websockets.connect(url, additional_headers=headers)
        
        
        
        await self.update_session({
            "modalities": ["text", "audio"],
            "instructions": self.instructions,
            "voice": self.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "whisper-1"},
            "turn_detection": {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 200},
            "temperature": self.temperature,
        })

    async def update_session(self, config: Dict[str, Any]) -> None:
        event = {"type": "session.update", "session": config}
        await self.ws.send(json.dumps(event))

    async def send_audio(self, audio_chunk: bytes):
        # print("Sending audio...")
        audio_b64 = base64.b64encode(audio_chunk).decode()
        # print(f"Audio length: {len(audio_chunk)}")
        await self.ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))

    async def handle_messages(self):
        print(f"Handling messages...")
        global audio_handler, request_tracker
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type")
                # print(f"Received event: {event}")  # Thêm dòng này

                if event_type == "conversation.item.created":
                    self._current_item_id = event.get("item", {}).get("id")
                    self._is_responding = True
                    # request_id = request_tracker.new_request()
                    # await request_tracker.track_request(request_id, self._current_item_id)

                elif event_type == "response.audio_transcript.done":
                    transcript = event.get("transcript", "")
                    response_id = event.get("item_id", "")
                    print(f"Transcript: {transcript}")
                    
                    # request_id = request_tracker.get_request_id(response_id)
                    # self._is_responding = False
                    # if request_id is not None:
                    #     await request_tracker.add_response(request_id, transcript)

        except websockets.exceptions.ConnectionClosed:
            print(f"Connection closed")
        except Exception as e:
            print(f"Error in message handling: {str(e)}")

    async def close(self) -> None:
        if self.ws:
            await self.ws.close()

class AudioHandler:
    def __init__(self):
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 24000
        self.chunk = 1024
        self.audio = pyaudio.PyAudio()
        self.streaming = False
        self.stream = None
        self.playback_stream = None
        self.playback_buffer = queue.Queue(maxsize=50)
        self.playback_event = threading.Event()
        self.playback_thread = None
        self.stop_playback = False

    async def start_streaming(self, worker: RealtimeClient):
        if self.streaming:
            return
        self.streaming = True
        self.stream = self.audio.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk)
        print(f"\nStreaming audio... Press 'q' to stop.")
        while self.streaming:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                await worker.send_audio(data)
            except Exception as e:
                logging.error(f"Streaming error: {e}")
                break
            await asyncio.sleep(0.05)

    def play_audio(self, audio_data: bytes):
        try:
            self.playback_buffer.put_nowait(audio_data)
        except queue.Full:
            self.playback_buffer.get_nowait()
            self.playback_buffer.put_nowait(audio_data)
        if not self.playback_thread or not self.playback_thread.is_alive():
            self.stop_playback = False
            self.playback_event.clear()
            self.playback_thread = threading.Thread(target=self._continuous_playback)
            self.playback_thread.start()

    def _continuous_playback(self):
        self.playback_stream = self.audio.open(format=self.format, channels=self.channels, rate=self.rate, output=True, frames_per_buffer=self.chunk)
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
            audio_segment = AudioSegment(audio_chunk, sample_width=2, frame_rate=24000, channels=1)
            audio_data = audio_segment.raw_data
            chunk_size = 24000
            for i in range(0, len(audio_data), chunk_size):
                if self.playback_event.is_set():
                    break
                chunk = audio_data[i:i+chunk_size]
                self.playback_stream.write(chunk)
        except Exception as e:
            print(f"Error playing audio chunk: {e}")

    def stop_streaming(self):
        self.streaming = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def stop_playback_immediately(self):
        self.stop_playback = True
        self.playback_buffer.queue.clear()
        self.playback_event.set()

    def cleanup(self):
        self.stop_playback_immediately()
        self.stop_playback = True
        if self.playback_thread:
            self.playback_thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

async def init_client():
    global worker, audio_handler, request_tracker
    request_tracker = RequestTracker()
    print("Initializing client...")
    audio_handler = AudioHandler()
    input_handler = InputHandler()
    input_handler.loop = asyncio.get_running_loop()

    worker = RealtimeClient(
        api_key=api_key,
        on_audio_delta=lambda audio: audio_handler.play_audio(audio),
        on_interrupt=lambda: audio_handler.stop_playback_immediately(),
        turn_detection_mode=TurnDetectionMode.SERVER_VAD
    )

    listener = keyboard.Listener(on_press=input_handler.on_press)
    listener.start()
    try:
        print("Connected to OpenAI Realtime API! Press 'q' to quit\n")
        await worker.connect()
        asyncio.create_task(worker.handle_messages())
        asyncio.create_task(audio_handler.start_streaming(worker))
        asyncio.create_task(process_responses())

        while True:
            command, _ = await input_handler.command_queue.get()
            if command == 'q':
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        audio_handler.stop_streaming()
        audio_handler.cleanup()
        await worker.close()

async def process_responses():
    global request_tracker
    while True:
        request_id, response = await request_tracker.get_next_response()
        print(f"Processing response {request_id}: {response}")
        logging.info(f"Processing response {request_id}: {response}")

if __name__ == "__main__":
    asyncio.run(init_client())