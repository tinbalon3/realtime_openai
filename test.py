import asyncio
import numpy as np
from agents import Agent
from agents.voice import VoicePipeline, SingleAgentVoiceWorkflow, WorkflowCallbacks
from utils import record_audio
from utils import AudioPlayer
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

class TranslatorCallbacks(WorkflowCallbacks):
    async def on_run(self, run):
        print(f"\n🗣️ You said: {run.input.transcript}")
        print(f"📝 Translation: {run.output.text}")

async def main():
    # Prompt rõ ràng
    translation_prompt = """
    You are a translation assistant.
    If the user speaks in English, translate it into Vietnamese.
    If the user speaks in Vietnamese, translate it into English.
    Do not explain anything. Only return the translated sentence.
    """

    agent = Agent(
        name="Translator",
        instructions=prompt_with_handoff_instructions(translation_prompt),
        model="gpt-4o-mini",
    )

    print("🎙️ Press SPACE to start recording, and press SPACE again to stop.")
    audio_data = await record_audio()

    pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent, callbacks=TranslatorCallbacks()))

    result = await pipeline.run(audio_data)

    with AudioPlayer() as player:
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                player.add_audio(event.data)

if __name__ == "__main__":
    asyncio.run(main())
