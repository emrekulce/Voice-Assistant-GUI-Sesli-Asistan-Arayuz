"""
========================================================================================
🎙️ Python AI Voice Assistant Floating HUD (Overlay GUI Template)
========================================================================================
A lightweight, modern, glassmorphic 60 FPS floating overlay GUI for AI Voice Assistants.
Zero GUI lag via isolated multiprocessing. Compatible with macOS, Windows, and Linux.

Works seamlessly with:
- Speech-to-Text (STT): Whisper, Vosk, Google Speech, Faster-Whisper, AssemblyAI
- Large Language Models (LLM): Gemini Live / REST, OpenAI Realtime / Chat, Claude, Ollama
- Text-to-Speech (TTS): ElevenLabs, Edge-TTS, Kokoro, gTTS, pyttsx3, Coqui-TTS

Author: Open Source Community
License: MIT
========================================================================================
"""

import enum
import math
import multiprocessing as mp
import os
import platform
import queue
import struct
import sys
import time
import tkinter as tk
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


# ======================================================================================
# ENUMS & CONFIGURATION
# ======================================================================================

class AssistantState(str, enum.Enum):
    """Supported state representations for the floating overlay GUI."""
    LISTENING = "listening"                  # Standby / Wake word detection / Ready
    USER_SPEAKING = "user_speaking"          # User microphone input active
    PROCESSING = "processing"                # LLM reasoning / Tool execution / API call
    ASSISTANT_SPEAKING = "assistant_speaking"# AI speech playback / TTS output
    SLEEPING = "sleeping"                    # Standby / Sleep / Muted mode
    ERROR = "error"                          # Exception / Connection failure / Alert


@dataclass
class OverlayConfig:
    """
    Configuration options for the Floating HUD Overlay.
    """
    width: int = 240
    height: int = 180
    margin_right: int = 24
    margin_top: int = 36
    border_radius: int = 20
    fps: int = 60
    always_on_top: bool = True
    draggable: bool = True
    opacity: float = 1.0
    title_font_family: str = "SF Pro Display"
    body_font_family: str = "SF Pro Text"
    initial_state: str = AssistantState.LISTENING.value
    custom_themes: Optional[Dict[str, Dict[str, str]]] = None


# Default Color & Theme Palettes
DEFAULT_THEMES: Dict[str, Dict[str, str]] = {
    AssistantState.LISTENING.value: {
        "primary": "#00E5FF",       # Vibrant Turquoise
        "secondary": "#0091EA",     # Deep Ocean Blue
        "glow": "#00E5FF",
        "bg": "#0D1117",
        "card_border": "#1F293D",
        "title": "Sıra Sizde",
        "status_text": "🎙️ Dinliyor...",
        "badge_color": "#00E5FF"
    },
    AssistantState.USER_SPEAKING.value: {
        "primary": "#D500F9",       # Neon Magenta
        "secondary": "#7C4DFF",     # Deep Violet
        "glow": "#E040FB",
        "bg": "#120B1F",
        "card_border": "#3A1D5A",
        "title": "Konuşuyorsunuz",
        "status_text": "👤 Dinleniyor...",
        "badge_color": "#E040FB"
    },
    AssistantState.PROCESSING.value: {
        "primary": "#FFAB00",       # Amber Glow
        "secondary": "#FF6D00",     # Warm Orange
        "glow": "#FFD600",
        "bg": "#16110A",
        "card_border": "#4A3614",
        "title": "İşlem Sürüyor",
        "status_text": "⚙️ Çalışıyor...",
        "badge_color": "#FFD600"
    },
    AssistantState.ASSISTANT_SPEAKING.value: {
        "primary": "#00F2FE",       # Aurora Cyan
        "secondary": "#FA709A",     # Electric Pink
        "glow": "#4FACFE",
        "bg": "#0E1222",
        "card_border": "#283256",
        "title": "Asistan Konuşuyor",
        "status_text": "🤖 Cevaplanıyor...",
        "badge_color": "#00F2FE"
    },
    AssistantState.SLEEPING.value: {
        "primary": "#78909C",       # Calm Slate
        "secondary": "#455A64",     # Muted Steel
        "glow": "#607D8B",
        "bg": "#0B0E14",
        "card_border": "#1E2530",
        "title": "Uyku Modu",
        "status_text": "💤 Uykuya Geçildi",
        "badge_color": "#78909C"
    },
    AssistantState.ERROR.value: {
        "primary": "#FF1744",       # Bright Crimson Red
        "secondary": "#D50000",     # Dark Crimson
        "glow": "#FF5252",
        "bg": "#1A0A0E",
        "card_border": "#4E121B",
        "title": "Hata Oluştu",
        "status_text": "⚠️ Bir sorun oluştu",
        "badge_color": "#FF1744"
    }
}


# ======================================================================================
# AUDIO UTILITIES
# ======================================================================================

def calculate_rms(
    audio_data: Union[bytes, bytearray, list, Any],
    sample_width: int = 2,
    max_expected: float = 8000.0
) -> float:
    """
    Computes a normalized Root Mean Square (RMS) volume (0.0 to 1.0) from raw PCM audio bytes
    or array-like numbers. Useful for feeding microphone or TTS stream amplitude to the HUD.

    :param audio_data: Raw PCM audio bytes, bytearray, or numpy array.
    :param sample_width: Byte width per sample (usually 2 for 16-bit PCM).
    :param max_expected: Sensitivity ceiling for RMS normalization (default 8000 for 16-bit).
    :return: Normalized float value clamped between 0.0 and 1.0.
    """
    if not audio_data:
        return 0.0

    try:
        # Check if numpy array
        if hasattr(audio_data, "__array__") or hasattr(audio_data, "shape"):
            import numpy as np
            arr = np.asarray(audio_data, dtype=np.float32)
            if arr.size == 0:
                return 0.0
            # If float in range [-1.0, 1.0]
            if np.max(np.abs(arr)) <= 1.0:
                arr = arr * 32767.0
            rms = float(np.sqrt(np.mean(np.square(arr))))
            return max(0.0, min(1.0, rms / max_expected))

        # Raw PCM 16-bit bytes
        if isinstance(audio_data, (bytes, bytearray)):
            if len(audio_data) < sample_width:
                return 0.0
            count = len(audio_data) // sample_width
            format_str = f"<{count}h" if sample_width == 2 else f"<{count}b"
            samples = struct.unpack(format_str, audio_data[:count * sample_width])
            sum_squares = sum(s * s for s in samples)
            mean_square = sum_squares / count
            rms = math.sqrt(mean_square)
            return max(0.0, min(1.0, rms / max_expected))

        # List of floats or integers
        if isinstance(audio_data, (list, tuple)):
            if not audio_data:
                return 0.0
            sum_squares = sum(float(x) * float(x) for x in audio_data)
            rms = math.sqrt(sum_squares / len(audio_data))
            if rms <= 1.0:
                return max(0.0, min(1.0, rms))
            return max(0.0, min(1.0, rms / max_expected))

    except Exception:
        pass
    return 0.0


# ======================================================================================
# CORE TKINTER OVERLAY ENGINE
# ======================================================================================

class AssistantOverlayGUI:
    """
    Glassmorphic Floating HUD Canvas Engine running 60 FPS procedural animations.
    """

    def __init__(self, cmd_queue: Optional[mp.Queue] = None, config: Optional[OverlayConfig] = None):
        self.cmd_queue = cmd_queue
        self.config = config or OverlayConfig()

        self.width = self.config.width
        self.height = self.config.height
        self.margin_right = self.config.margin_right
        self.margin_top = self.config.margin_top
        self.border_radius = self.config.border_radius

        # Themes
        self.themes = dict(DEFAULT_THEMES)
        if self.config.custom_themes:
            self.themes.update(self.config.custom_themes)

        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self.is_running = False

        # State Management
        self.state = str(self.config.initial_state)
        self.tool_detail = ""
        self.transcript_snippet = ""
        self.user_volume = 0.0       # Smoothed 0.0 - 1.0
        self.assistant_volume = 0.0  # Smoothed 0.0 - 1.0
        self.target_user_vol = 0.0
        self.target_asst_vol = 0.0

        # Animation Ticks
        self.anim_tick = 0.0
        self.spin_angle = 0.0
        self.state_transition = 1.0
        self.last_state = self.state

        # Drag-and-drop offsets
        self._drag_start_x = 0
        self._drag_start_y = 0

    def start(self):
        """Initializes the Tkinter window and starts the event loop."""
        self.root = tk.Tk()
        self.root.title("AI Assistant HUD")

        # Configure Cross-Platform Transparency & Window Styling
        self._setup_window_styling()

        # Initial Positioning
        self._position_window()

        # Canvas Construction
        transparent_bg = "systemTransparent" if platform.system() == "Darwin" else (
            "#010101" if platform.system() == "Windows" else "#0D1117"
        )
        self.canvas = tk.Canvas(
            self.root,
            width=self.width,
            height=self.height,
            bg=transparent_bg,
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        # Drag and Drop Bindings
        if self.config.draggable:
            self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
            self.canvas.bind("<B1-Motion>", self._on_drag_motion)
            self.canvas.bind("<Double-Button-1>", self._reset_position)

        self.is_running = True
        self._schedule_next_frame()
        self.root.mainloop()

    def _setup_window_styling(self):
        """Applies OS-specific transparent window attributes."""
        if not self.root:
            return

        current_os = platform.system()
        self.root.overrideredirect(True)

        if self.config.always_on_top:
            try:
                self.root.wm_attributes("-topmost", True)
            except Exception:
                pass

        if current_os == "Darwin":  # macOS
            try:
                self.root.wm_attributes("-transparent", True)
                self.root.config(bg="systemTransparent")
            except Exception:
                pass
        elif current_os == "Windows":  # Windows
            try:
                # Key out #010101 color as transparent
                self.root.config(bg="#010101")
                self.root.wm_attributes("-transparentcolor", "#010101")
                if 0.0 < self.config.opacity < 1.0:
                    self.root.wm_attributes("-alpha", self.config.opacity)
            except Exception:
                pass
        else:  # Linux / X11
            try:
                if 0.0 < self.config.opacity < 1.0:
                    self.root.wm_attributes("-alpha", self.config.opacity)
            except Exception:
                pass

    def _position_window(self):
        """Positions window at top-right corner of screen."""
        if not self.root:
            return
        screen_w = self.root.winfo_screenwidth()
        x_pos = screen_w - self.width - self.margin_right
        y_pos = self.margin_top
        self.root.geometry(f"{self.width}x{self.height}+{x_pos}+{y_pos}")

    def _on_drag_start(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_drag_motion(self, event):
        if not self.root:
            return
        x = self.root.winfo_x() - self._drag_start_x + event.x
        y = self.root.winfo_y() - self._drag_start_y + event.y
        self.root.geometry(f"+{x}+{y}")

    def _reset_position(self, event=None):
        self._position_window()

    def _schedule_next_frame(self):
        """Animation loop with target FPS."""
        if not self.is_running or not self.root:
            return

        if not self._process_queue():
            return

        try:
            self._update_animation()
            self._render_frame()
        except (tk.TclError, Exception):
            return

        if self.is_running and self.root:
            frame_ms = max(10, int(1000 / max(1, self.config.fps)))
            try:
                self.root.after(frame_ms, self._schedule_next_frame)
            except (tk.TclError, Exception):
                pass

    def _process_queue(self) -> bool:
        """Processes IPC queue commands. Returns False if requested to stop."""
        if not self.cmd_queue:
            return True

        while True:
            try:
                cmd, data = self.cmd_queue.get_nowait()
                if cmd == "set_state":
                    new_state, detail = data
                    if new_state != self.state:
                        self.last_state = self.state
                        self.state = new_state
                        self.state_transition = 0.0
                    self.tool_detail = detail
                elif cmd == "set_user_vol":
                    self.target_user_vol = max(0.0, min(1.0, float(data)))
                elif cmd == "set_asst_vol":
                    self.target_asst_vol = max(0.0, min(1.0, float(data)))
                elif cmd == "set_transcript":
                    self.transcript_snippet = str(data)
                elif cmd == "close":
                    self.is_running = False
                    try:
                        self.root.destroy()
                    except Exception:
                        pass
                    return False
            except (queue.Empty, EOFError):
                break
            except Exception:
                break
        return self.is_running

    def _update_animation(self):
        """Smoothly steps animation ticks and interpolates audio RMS."""
        self.anim_tick += 0.05
        self.spin_angle = (self.spin_angle + 4.0) % 360.0

        # Low-pass filter for volume levels (smooth transitions)
        self.user_volume = self.user_volume * 0.7 + self.target_user_vol * 0.3
        self.assistant_volume = self.assistant_volume * 0.7 + self.target_asst_vol * 0.3

        if self.state_transition < 1.0:
            self.state_transition = min(1.0, self.state_transition + 0.08)

    def _render_frame(self):
        """Draws procedural visualizer and status elements onto Tkinter canvas."""
        if not self.is_running or not self.canvas:
            return
        self.canvas.delete("all")

        theme = self.themes.get(self.state, self.themes.get(AssistantState.LISTENING.value, DEFAULT_THEMES["listening"]))
        w, h = self.width, self.height
        radius = self.border_radius

        # 1. Glassmorphism Card Background
        self._draw_rounded_rect(
            4, 4, w - 4, h - 4,
            radius=radius,
            fill=theme["bg"],
            outline=theme["card_border"],
            width=1.5
        )

        # Top Accent Glow Line
        self.canvas.create_line(
            radius + 10, 4, w - radius - 10, 4,
            fill=theme["primary"],
            width=2
        )

        # 2. Central Dynamic Visualizer (Orb / Waves / Spinner)
        orb_cx = w / 2.0
        orb_cy = 68.0

        if self.state == AssistantState.LISTENING.value:
            # Calming breathing pulse
            pulse = (math.sin(self.anim_tick * 1.5) + 1.0) / 2.0
            r1 = 24 + pulse * 6
            r2 = 17 + pulse * 3

            self.canvas.create_oval(orb_cx - r1, orb_cy - r1, orb_cx + r1, orb_cy + r1,
                                    outline=theme["secondary"], width=1.5)
            self.canvas.create_oval(orb_cx - r2, orb_cy - r2, orb_cx + r2, orb_cy + r2,
                                    fill=theme["primary"], outline="")
            self.canvas.create_oval(orb_cx - 8, orb_cy - 8, orb_cx + 8, orb_cy + 8,
                                    fill="#FFFFFF", outline="")

        elif self.state == AssistantState.USER_SPEAKING.value:
            # Dynamic reactive microphone equalizer & expanding aura
            vol_boost = self.user_volume * 28.0
            r_outer = 22 + vol_boost
            r_mid = 16 + vol_boost * 0.7
            r_core = 11 + vol_boost * 0.4

            self.canvas.create_oval(orb_cx - r_outer, orb_cy - r_outer, orb_cx + r_outer, orb_cy + r_outer,
                                    outline=theme["glow"], width=2.0)
            self.canvas.create_oval(orb_cx - r_mid, orb_cy - r_mid, orb_cx + r_mid, orb_cy + r_mid,
                                    fill=theme["secondary"], outline="")
            self.canvas.create_oval(orb_cx - r_core, orb_cy - r_core, orb_cx + r_core, orb_cy + r_core,
                                    fill=theme["primary"], outline="")

            # 7-Band Equalizer
            bar_count = 7
            spacing = 10
            start_x = orb_cx - (bar_count - 1) * spacing / 2.0
            for i in range(bar_count):
                wave_phase = math.sin(self.anim_tick * 4.0 + i * 0.8)
                bar_h = max(4.0, (self.user_volume * 35.0 + 5.0) * (0.5 + 0.5 * wave_phase))
                bx = start_x + i * spacing
                self.canvas.create_line(bx, orb_cy - bar_h / 2, bx, orb_cy + bar_h / 2,
                                        fill="#FFFFFF", width=2.5, capstyle="round")

        elif self.state == AssistantState.PROCESSING.value:
            # Orbital thinking spinner & glowing amber core
            r_spin = 24
            rad = math.radians(self.spin_angle)

            for i in range(4):
                ang = rad + i * (math.pi / 2.0)
                px = orb_cx + r_spin * math.cos(ang)
                py = orb_cy + r_spin * math.sin(ang)
                dot_r = 3.5 if (i % 2 == 0) else 2.0
                self.canvas.create_oval(px - dot_r, py - dot_r, px + dot_r, py + dot_r,
                                        fill=theme["primary"] if i % 2 == 0 else theme["secondary"], outline="")

            self.canvas.create_oval(orb_cx - r_spin, orb_cy - r_spin, orb_cx + r_spin, orb_cy + r_spin,
                                    outline=theme["card_border"], width=1.5)

            core_pulse = (math.sin(self.anim_tick * 3.0) + 1.0) / 2.0
            core_r = 11 + core_pulse * 4
            self.canvas.create_oval(orb_cx - core_r, orb_cy - core_r, orb_cx + core_r, orb_cy + core_r,
                                    fill=theme["primary"], outline="")

        elif self.state == AssistantState.ASSISTANT_SPEAKING.value:
            # 9-Band Aurora frequency spectrum for speech output
            vol_boost = self.assistant_volume * 22.0
            r_glow = 25 + vol_boost

            self.canvas.create_oval(orb_cx - r_glow, orb_cy - r_glow, orb_cx + r_glow, orb_cy + r_glow,
                                    outline=theme["secondary"], width=1.5)

            bar_count = 9
            spacing = 9
            start_x = orb_cx - (bar_count - 1) * spacing / 2.0
            for i in range(bar_count):
                wave = math.sin(self.anim_tick * 5.0 + i * 0.7)
                height_factor = math.cos((i - (bar_count // 2)) / (bar_count / 2.0) * (math.pi / 2.5))
                bar_h = max(5.0, (self.assistant_volume * 40.0 + 8.0) * height_factor * (0.6 + 0.4 * wave))
                bx = start_x + i * spacing
                color = theme["primary"] if i % 2 == 0 else theme["secondary"]
                self.canvas.create_line(bx, orb_cy - bar_h / 2, bx, orb_cy + bar_h / 2,
                                        fill=color, width=2.5, capstyle="round")

        elif self.state == AssistantState.SLEEPING.value:
            # Sleep / Muted ring
            self.canvas.create_oval(orb_cx - 16, orb_cy - 16, orb_cx + 16, orb_cy + 16,
                                    fill=theme["secondary"], outline=theme["card_border"])
            self.canvas.create_oval(orb_cx - 8, orb_cy - 8, orb_cx + 8, orb_cy + 8,
                                    fill=theme["primary"], outline="")

        elif self.state == AssistantState.ERROR.value:
            # Crimson alert pulse
            alert_pulse = (math.sin(self.anim_tick * 6.0) + 1.0) / 2.0
            r_err = 20 + alert_pulse * 6
            self.canvas.create_oval(orb_cx - r_err, orb_cy - r_err, orb_cx + r_err, orb_cy + r_err,
                                    outline=theme["glow"], width=2.0)
            self.canvas.create_oval(orb_cx - 14, orb_cy - 14, orb_cx + 14, orb_cy + 14,
                                    fill=theme["primary"], outline="")
            # Exclamation mark
            self.canvas.create_text(orb_cx, orb_cy, text="!", fill="#FFFFFF", font=(self.config.title_font_family, 14, "bold"))

        # 3. Title & Subtext Badge
        self.canvas.create_text(
            orb_cx, 122,
            text=theme["title"],
            fill="#FFFFFF",
            font=(self.config.title_font_family, 13, "bold"),
            anchor="center"
        )

        subtext = self.tool_detail if self.tool_detail else theme["status_text"]
        if self.transcript_snippet and self.state in [AssistantState.USER_SPEAKING.value, AssistantState.ASSISTANT_SPEAKING.value]:
            subtext = self.transcript_snippet

        # Clip overly long text for badge
        if len(subtext) > 30:
            subtext = subtext[:28] + "..."

        # Capsule Pill
        self._draw_rounded_rect(
            18, 140, w - 18, 166,
            radius=12,
            fill=theme["card_border"],
            outline="",
            width=0
        )

        # Status Dot
        self.canvas.create_oval(
            27, 149, 35, 157,
            fill=theme["badge_color"],
            outline=""
        )

        # Subtext Label
        self.canvas.create_text(
            42, 153,
            text=subtext,
            fill="#E0E6ED",
            font=(self.config.body_font_family, 10),
            anchor="w"
        )

    def _draw_rounded_rect(self, x1, y1, x2, y2, radius=20, **kwargs):
        """Helper to draw smooth antialiased rounded rectangles on Tkinter Canvas."""
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1
        ]
        return self.canvas.create_polygon(points, smooth=True, **kwargs)


# ======================================================================================
# MULTIPROCESSING CONTROLLER & API
# ======================================================================================

def _gui_process_worker(cmd_queue: mp.Queue, config: OverlayConfig):
    """Entry point for GUI loop running in dedicated child process."""
    gui = AssistantOverlayGUI(cmd_queue=cmd_queue, config=config)
    gui.start()


class OverlayController:
    """
    Lightweight, thread-safe controller to interact with the floating HUD.
    """

    def __init__(self, config: Optional[OverlayConfig] = None):
        self.config = config or OverlayConfig()
        self.cmd_queue = mp.Queue()
        self.process = mp.Process(
            target=_gui_process_worker,
            args=(self.cmd_queue, self.config),
            daemon=True
        )
        self.process.start()

    # --- Core Controls ---

    def set_state(self, state: Union[AssistantState, str], detail: str = ""):
        """
        Sets the visual state of the HUD overlay.
        :param state: 'listening', 'user_speaking', 'processing', 'assistant_speaking', 'sleeping', 'error'
        :param detail: Optional status/subtext to show in the bottom pill badge.
        """
        state_val = state.value if isinstance(state, AssistantState) else str(state)
        try:
            self.cmd_queue.put(("set_state", (state_val, detail)))
        except Exception:
            pass

    def update_user_volume(self, rms_volume: float):
        """Updates user speaking amplitude for equalizer animation (0.0 to 1.0)."""
        try:
            self.cmd_queue.put(("set_user_vol", float(rms_volume)))
        except Exception:
            pass

    def update_assistant_volume(self, rms_volume: float):
        """Updates assistant speech amplitude for Aurora spectrum animation (0.0 to 1.0)."""
        try:
            self.cmd_queue.put(("set_asst_vol", float(rms_volume)))
        except Exception:
            pass

    def set_transcript(self, text: str):
        """Displays real-time partial or full transcript in the badge."""
        try:
            self.cmd_queue.put(("set_transcript", str(text)))
        except Exception:
            pass

    def stop(self):
        """Gracefully closes the HUD overlay window and stops the background process."""
        try:
            self.cmd_queue.put(("close", None))
            time.sleep(0.08)
            if self.process.is_alive():
                self.process.terminate()
        except Exception:
            pass

    def is_alive(self) -> bool:
        """Returns True if the background GUI process is still running."""
        return self.process.is_alive()

    # --- Convenient Helper Shortcuts ---

    def listening(self, detail: str = "🎙️ Dinliyor..."):
        """Shortcut: Switch to Listening state."""
        self.set_state(AssistantState.LISTENING, detail)

    def user_speaking(self, detail: str = "👤 Dinleniyor...", volume: float = 0.0, transcript: str = ""):
        """Shortcut: Switch to User Speaking state with optional volume and transcript."""
        self.set_state(AssistantState.USER_SPEAKING, detail)
        if volume > 0:
            self.update_user_volume(volume)
        if transcript:
            self.set_transcript(transcript)

    def processing(self, detail: str = "⚙️ Çalışıyor..."):
        """Shortcut: Switch to Processing / AI Reasoning state."""
        self.set_state(AssistantState.PROCESSING, detail)

    def assistant_speaking(self, detail: str = "🤖 Cevaplanıyor...", volume: float = 0.0, transcript: str = ""):
        """Shortcut: Switch to Assistant Speaking state with optional volume and transcript."""
        self.set_state(AssistantState.ASSISTANT_SPEAKING, detail)
        if volume > 0:
            self.update_assistant_volume(volume)
        if transcript:
            self.set_transcript(transcript)

    def sleeping(self, detail: str = "💤 Uyku Modu"):
        """Shortcut: Switch to Sleep / Standby state."""
        self.set_state(AssistantState.SLEEPING, detail)

    def error(self, detail: str = "⚠️ Hata"):
        """Shortcut: Switch to Error alert state."""
        self.set_state(AssistantState.ERROR, detail)

    # --- Context Manager Support ---

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ======================================================================================
# EXTENSIBLE BASE ASSISTANT BRIDGE
# ======================================================================================

class BaseAssistantBridge:
    """
    Modular Bridge class designed for developers to inherit or hook into custom AI assistants.
    Provides lifecycle event handlers that update the HUD state automatically.
    """

    def __init__(self, overlay: Optional[OverlayController] = None, config: Optional[OverlayConfig] = None):
        self.overlay = overlay or start_overlay(config=config)

    def on_idle(self, detail: str = "🎙️ Dinliyor..."):
        """Called when assistant is in standby or listening for wake word."""
        if self.overlay:
            self.overlay.listening(detail)

    def on_wake_word_detected(self, wake_word: str = ""):
        """Called when a wake word is detected."""
        msg = f"🎙️ '{wake_word}' algılandı..." if wake_word else "🎙️ Sizi dinliyor..."
        if self.overlay:
            self.overlay.listening(msg)

    def on_user_speech_start(self, detail: str = "👤 Dinleniyor..."):
        """Called when user starts talking into the microphone."""
        if self.overlay:
            self.overlay.user_speaking(detail)

    def on_user_speech_chunk(self, audio_rms_volume: float = 0.0, partial_transcript: str = ""):
        """Called for every microphone audio chunk during user speech."""
        if self.overlay:
            if audio_rms_volume >= 0.0:
                self.overlay.update_user_volume(audio_rms_volume)
            if partial_transcript:
                self.overlay.set_transcript(partial_transcript)

    def on_processing_start(self, action_description: str = "⚙️ Düşünüyor..."):
        """Called when LLM request or Tool execution starts."""
        if self.overlay:
            self.overlay.processing(action_description)

    def on_assistant_speech_start(self, full_response_text: str = "", detail: str = "🤖 Cevaplanıyor..."):
        """Called when TTS playback begins."""
        if self.overlay:
            self.overlay.assistant_speaking(detail)
            if full_response_text:
                self.overlay.set_transcript(full_response_text)

    def on_assistant_speech_chunk(self, audio_rms_volume: float = 0.0, chunk_text: str = ""):
        """Called for each TTS audio chunk played to speakers."""
        if self.overlay:
            if audio_rms_volume >= 0.0:
                self.overlay.update_assistant_volume(audio_rms_volume)
            if chunk_text:
                self.overlay.set_transcript(chunk_text)

    def on_error(self, error_message: str = "⚠️ Bir hata oluştu"):
        """Called on connection or runtime error."""
        if self.overlay:
            self.overlay.error(error_message)

    def on_sleep(self, detail: str = "💤 Uyku Modu"):
        """Called when assistant enters sleep mode."""
        if self.overlay:
            self.overlay.sleeping(detail)

    def close(self):
        """Closes the overlay HUD."""
        if self.overlay:
            self.overlay.stop()


# ======================================================================================
# SINGLETON INSTANCE & CONVENIENCE FUNCTIONS
# ======================================================================================

_controller_instance: Optional[OverlayController] = None


def start_overlay(config: Optional[OverlayConfig] = None) -> OverlayController:
    """
    Launches or returns the singleton HUD Overlay instance.
    :param config: Optional OverlayConfig instance for customization.
    :return: OverlayController instance.
    """
    global _controller_instance
    if _controller_instance is None or not _controller_instance.is_alive():
        _controller_instance = OverlayController(config=config)
    return _controller_instance


def get_overlay() -> Optional[OverlayController]:
    """Returns the currently active OverlayController instance if any."""
    return _controller_instance


# ======================================================================================
# STANDALONE DEMO & TEST RUNNER
# ======================================================================================

if __name__ == "__main__":
    print("==========================================================")
    print("🚀 AI Voice Assistant Overlay GUI - Live Interactive Demo")
    print("==========================================================")
    print("Testing animation states & procedural audio visualizers...")
    print("Press Ctrl+C in terminal to stop.")

    demo_config = OverlayConfig(
        width=240,
        height=180,
        margin_right=24,
        margin_top=36,
        always_on_top=True,
        draggable=True
    )

    overlay = start_overlay(config=demo_config)

    demo_states = [
        ("listening", "🎙️ Sizi Dinliyor...", 0.0),
        ("user_speaking", "👤 Kullanıcı konuşuyor...", 0.7),
        ("processing", "⚙️ Bilgiler Taranıyor...", 0.0),
        ("processing", "💻 Kod Üretiliyor...", 0.0),
        ("assistant_speaking", "🤖 Asistan cevap veriyor...", 0.8),
        ("error", "⚠️ Bağlantı Zaman Aşımı", 0.0),
        ("sleeping", "💤 Uykuya Geçildi", 0.0),
    ]

    try:
        while True:
            for s_name, detail, vol in demo_states:
                print(f"▶ State: {s_name} | {detail}")
                overlay.set_state(s_name, detail)

                if s_name == "user_speaking":
                    for i in range(30):
                        simulated_vol = 0.3 + 0.6 * abs(math.sin(i * 0.35))
                        overlay.update_user_volume(simulated_vol)
                        overlay.set_transcript(f"Hava bugün nasıl? ({i+1})")
                        time.sleep(0.08)
                elif s_name == "assistant_speaking":
                    for i in range(35):
                        simulated_vol = 0.4 + 0.5 * abs(math.cos(i * 0.45))
                        overlay.update_assistant_volume(simulated_vol)
                        overlay.set_transcript("Bugün hava güneşli ve 24 derece.")
                        time.sleep(0.08)
                else:
                    time.sleep(2.2)
    except KeyboardInterrupt:
        print("\nStopping HUD Demo...")
    finally:
        overlay.stop()
        print("HUD Closed successfully.")
