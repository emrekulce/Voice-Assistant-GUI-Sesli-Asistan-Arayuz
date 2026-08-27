"""
========================================================================================
Example: Custom Voice Assistant using Floating HUD Overlay Template
========================================================================================
This example demonstrates how to integrate your own Speech-to-Text (STT),
Large Language Model (LLM), and Text-to-Speech (TTS) logic using the
provided BaseAssistantBridge and OverlayController.
========================================================================================
"""

import time
import math
import random
from assistant_overlay_gui import (
    start_overlay,
    BaseAssistantBridge,
    OverlayConfig,
    AssistantState
)


class MyCustomVoiceAssistant(BaseAssistantBridge):
    """
    Example custom voice assistant implementation.
    Replace the mock methods below with your actual STT / LLM / TTS libraries!
    """

    def listen_and_process_cycle(self):
        """Simulates one complete interactive turn of the voice assistant."""

        # 1. STANDBY / LISTENING FOR WAKE WORD
        print("1. Standby: Listening for wake word...")
        self.on_wake_word_detected()
        time.sleep(1.5)

        # 2. USER STARTS SPEAKING (Simulating microphone input)
        print("2. User speaking...")
        self.on_user_speech_start()

        user_query = "Hava durumu bugün nasıl?"
        for i in range(25):
            # Dynamic simulated RMS audio volume (0.0 - 1.0)
            mock_volume = 0.3 + 0.6 * abs(math.sin(i * 0.4))
            self.on_user_speech_chunk(audio_rms_volume=mock_volume, partial_transcript=user_query[:i+1])
            time.sleep(0.08)

        # 3. PROCESSING / AI REASONING / TOOL CALLING
        print("3. Processing: AI reasoning & Tool execution...")
        self.on_processing_start(action_description="🌦️ Hava Durumu API Sorgulanıyor...")
        time.sleep(1.8)

        # 4. ASSISTANT SPEECH / TTS PLAYBACK
        print("4. Assistant speaking: TTS playback...")
        ai_response = "Bugün İstanbul'da hava parçalı bulutlu ve 22 derece."
        self.on_assistant_speech_start(full_response_text=ai_response)

        for i in range(35):
            # Dynamic simulated speech audio volume
            mock_volume = 0.4 + 0.5 * abs(math.cos(i * 0.45))
            self.on_assistant_speech_chunk(audio_rms_volume=mock_volume)
            time.sleep(0.08)

        # 5. BACK TO IDLE / STANDBY
        print("5. Completed turn, back to standby.\n")
        self.on_idle()
        time.sleep(2.0)


def main():
    # 1. Optional custom configuration
    config = OverlayConfig(
        width=240,
        height=180,
        margin_right=30,
        margin_top=40,
        always_on_top=True,
        draggable=True
    )

    # 2. Start the floating HUD
    hud = start_overlay(config=config)

    # 3. Create your assistant instance
    assistant = MyCustomVoiceAssistant(overlay=hud)

    try:
        print("🚀 Starting Assistant Loop (Press Ctrl+C to stop)...")
        # Run a couple of conversation turns
        for turn in range(2):
            print(f"--- Turn {turn + 1} ---")
            assistant.listen_and_process_cycle()

        # Demonstrate sleep mode
        print("Putting assistant to sleep...")
        assistant.on_sleep()
        time.sleep(2.5)

    except KeyboardInterrupt:
        print("\nStopping assistant...")
    finally:
        hud.stop()
        print("HUD closed.")


if __name__ == "__main__":
    main()
