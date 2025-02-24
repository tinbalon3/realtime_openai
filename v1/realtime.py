import base64
from enum import Enum
import json
from typing import Any, Callable, Dict, Optional
import websockets
from v1.realtime_client import TurnDetectionMode
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
        print("Sending audio...")
        audio_b64 = base64.b64encode(audio_chunk).decode()
        await self.ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))

    async def handle_messages(self):
        """Xử lý tin nhắn từ WebSocket với đồng bộ hóa trạng thái."""
        print(f"Handling messages for {self}")
        global audio_handler, request_tracker,workers
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type")

                if event_type == "conversation.item.created":
                    self._current_item_id = event.get("item", {}).get("id")
                    self._is_responding = True
                    request_id = request_tracker.new_request()
                    await request_tracker.track_request(request_id, self._current_item_id)
                    worker = next((w for w in workers if w.worker == self), None)
                    if worker:
                        await worker.set_status(False, True, worker.available_Used)

                elif event_type == "response.done":
                    worker = next((w for w in workers if w.worker == self), None)
                    if worker:
                        await worker.set_status(True, False, worker.available_Used)

                elif event_type == "response.audio_transcript.done":
                    transcript = event.get("transcript", "")
                    response_id = event.get("item_id", "")
                    request_id = request_tracker.get_request_id(response_id)
                    self._is_responding = False
                    worker = next((w for w in workers if w.worker == self), None)
                    if worker:
                        await worker.set_status(True, False, worker.available_Used)
                    if request_id is not None:
                        await request_tracker.add_response(request_id, transcript)

        except websockets.exceptions.ConnectionClosed:
            print(f"Connection closed for {self}")
        except Exception as e:
            print(f"Error in message handling: {str(e)}")
    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self.ws:
            await self.ws.close()