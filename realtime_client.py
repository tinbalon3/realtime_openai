import asyncio
import websockets
import json
import base64
import io

from typing import Optional, Callable, List, Dict, Any
from enum import Enum
from pydub import AudioSegment




class TurnDetectionMode(Enum):
    SERVER_VAD = "server_vad"
    MANUAL = "manual"

class RealtimeClient:
   
    def __init__(
        self, 
        api_key: str,
        model: str = "gpt-4o-realtime-preview-2024-10-01",
        voice: str = "alloy",
        instructions = """
        You are a translation tool. Your sole task is to read text in English and provide an accurate, concise Vietnamese translation. 
        Do not offer explanations, clarifications, or any additional responses besides the translated text.
        If you are not able to understand the text to translate, don’t return anything, remember if dont understand , dont return the text.
                        """,
        temperature: float = 0.8,
        turn_detection_mode: TurnDetectionMode = TurnDetectionMode.MANUAL,
        on_text_delta: Optional[Callable[[str], None]] = None,
        on_audio_delta: Optional[Callable[[bytes], None]] = None,
        on_interrupt: Optional[Callable[[], None]] = None,
        on_input_transcript: Optional[Callable[[str], None]] = None,  
        on_output_transcript: Optional[Callable[[str], None]] = None,  
        extra_event_handlers: Optional[Dict[str, Callable[[Dict[str, Any]], None]]] = None,
        response_queue = None,
        name = None,
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
        self.name = name
        self.tools  = []

        # Track current response state
        self._current_response_id = None
        self._current_item_id = None
        self._is_responding = False
        # Track printing state for input and output transcripts
        self._print_input_transcript = False
        self._output_transcript_buffer = ""
        
        self.response_queue = response_queue
        
        

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
                    "prefix_padding_ms": 500,
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

    async def send_text(self, text: str) -> None:
        """Send text message to the API."""
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": text
                }]
            }
        }
        await self.ws.send(json.dumps(event))
        await self.create_response()

    async def send_audio(self, audio_bytes: bytes) -> None:
        """Send audio data to the API."""
        # Convert audio to required format (24kHz, mono, PCM16)
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_frame_rate(24000).set_channels(1).set_sample_width(2)
        pcm_data = base64.b64encode(audio.raw_data).decode()
        
        # Append audio to buffer
        append_event = {
            "type": "input_audio_buffer.append",
            "audio": pcm_data
        }
        await self.ws.send(json.dumps(append_event))
        
        # Commit the buffer
        commit_event = {
            "type": "input_audio_buffer.commit"
        }
        await self.ws.send(json.dumps(commit_event))
        
        # In manual mode, we need to explicitly request a response
        if self.turn_detection_mode == TurnDetectionMode.MANUAL:
            await self.create_response()

    async def stream_audio(self, audio_chunk: bytes) -> None:
        """Stream raw audio data to the API."""
        audio_b64 = base64.b64encode(audio_chunk).decode()
        
        append_event = {
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }
        await self.ws.send(json.dumps(append_event))

    async def create_response(self, functions: Optional[List[Dict[str, Any]]] = None) -> None:
        """Request a response from the API. Needed when using manual mode."""
        event = {
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"]
            }
        }
        
        if functions:
            event["response"]["tools"] = functions
            
        await self.ws.send(json.dumps(event))

    async def send_function_result(self, call_id: str, result: Any) -> None:
        """Send function call result back to the API."""
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result
            }
        }
        await self.ws.send(json.dumps(event))

        # functions need a manual response
        await self.create_response()

    async def cancel_response(self) -> None:
        """Cancel the current response."""
        event = {
            "type": "response.cancel"
        }
        await self.ws.send(json.dumps(event))
    
    async def truncate_response(self):
        """Truncate the conversation item to match what was actually played."""
        if self._current_item_id:
            event = {
                "type": "conversation.item.truncate",
                "item_id": self._current_item_id
            }
            await self.ws.send(json.dumps(event))

   

    # async def handle_interruption(self):
    #     """Handle user interruption of the current response."""
    #     if not self._is_responding:
    #         return
            
    #     print("\n[Handling interruption]")
        
    #     # 1. Cancel the current response
    #     if self._current_response_id:
    #         await self.cancel_response()
        
    #     # 2. Truncate the conversation item to what was actually played
    #     if self._current_item_id:
    #         await self.truncate_response()
            
    #     self._is_responding = False
    #     self._current_response_id = None
    #     self._current_item_id = None

    async def handle_messages(self) -> None:
        print("Handling messages...")
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type")
                
                # Handle error events
                if event_type == "error":
                    print(f"Error: {event['error']}")
                    continue
                
                # Handle session updated events
                elif event_type == "session.updated":
                    pass
                
                # Track response state when a response is created
                elif event_type == "response.created":
                    self._current_response_id = event.get("response", {}).get("id")
                    self._is_responding = True
                
                # Track response item added events
                elif event_type == "response.output_item.added":
                    self._current_item_id = event.get("item", {}).get("id") 
                    
                # Handle response completion events
                elif event_type == "response.done":
                    self._is_responding = False
                    self._current_response_id = None
                    self._current_item_id = None
                    
                    
                # Handle speech started events
                elif event_type == "input_audio_buffer.speech_started":
                    print(f"\n[Speech started]")
                    
                    
                
                # Handle speech stopped events
                elif event_type == "input_audio_buffer.speech_stopped":
                    print("\n[Speech ended]")
                    
                
                
                # Handle text delta events
                elif event_type == "response.text.delta":
                    print("Text delta:", event["delta"])
                    if self.on_text_delta:
                        self.on_text_delta(event["delta"])
                        
                # Handle audio delta events
                elif event_type == "response.audio.delta":
                    if self.on_audio_delta:
                        audio_bytes = base64.b64decode(event["delta"])
                        self.on_audio_delta(audio_bytes)
                        
                # Handle function call arguments completion
                elif event_type == "response.function_call_arguments.done":
                    await self.call_tool(event["call_id"], event['name'], json.loads(event['arguments']))

                # Handle input audio transcription completion
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "")
                    if self.on_input_transcript:
                        await asyncio.to_thread(self.on_input_transcript, transcript)
                        self._print_input_transcript = True

                # Handle output audio transcription delta
                elif event_type == "response.audio_transcript.delta":
                    if self.on_output_transcript:
                        delta = event.get("delta", "")
                        if not self._print_input_transcript:
                            self._output_transcript_buffer += delta
                        else:
                            if self._output_transcript_buffer:
                                await asyncio.to_thread(self.on_output_transcript, self._output_transcript_buffer)
                                self._output_transcript_buffer = ""
                            await asyncio.to_thread(self.on_output_transcript, delta)

                # Handle output audio transcription completion
                elif event_type == "response.audio_transcript.done":
                    # if self.on_text_delta:
                    #     self._is_responding = False
                    await self.response_queue.put((event["transcript"],self.name))
                    
                        # self.on_text_delta(event["transcript"])
                    self._print_input_transcript = False

                # Handle custom extra event handlers
                elif event_type in self.extra_event_handlers:
                    self.extra_event_handlers[event_type](event)

        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")
        except Exception as e:
            print(f"Error in message handling: {str(e)}")

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self.ws:
            await self.ws.close()