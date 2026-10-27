"""
Voice & Audio Summarization module supporting English, Hindi (हिंदी), and Marathi (मराठी).
Provides real-time browser speech recognition, audio file transcription, and Text-to-Speech.
"""

import json
import urllib.request
import urllib.error
import io
import wave
from typing import Tuple, Optional


# Supported Speech Recognition Languages
SUPPORTED_VOICE_LANGUAGES = {
    "en": {"code": "en-IN", "name": "English (India / Global)"},
    "hi": {"code": "hi-IN", "name": "Hindi (हिंदी)"},
    "mr": {"code": "mr-IN", "name": "Marathi (मराठी)"},
}


def transcribe_wav_audio_bytes(audio_bytes: bytes, language_code: str = "en-IN") -> Tuple[bool, str]:
    """
    Transcribe WAV audio bytes via Google Speech Recognition endpoint.
    Requires no heavy external dependencies; uses standard Python urllib.
    """
    if not audio_bytes:
        return False, "Audio recording is empty."

    url = f"https://www.google.com/speech-api/v2/recognize?output=json&lang={language_code}&key=&client=chromium"
    headers = {"Content-Type": "audio/l16; rate=16000"}

    try:
        # Check if valid WAV
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                framerate = wf.getframerate()
                headers["Content-Type"] = f"audio/l16; rate={framerate}"
                raw_frames = wf.readframes(wf.getnframes())
                payload = raw_frames
        except Exception:
            # Fallback to direct raw bytes
            payload = audio_bytes

        req = urllib.request.Request(url, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8")
            # Google Speech returns newline-separated JSON chunks
            for line in content.splitlines():
                if line.strip():
                    data = json.loads(line)
                    result = data.get("result", [])
                    if result and len(result) > 0:
                        alternatives = result[0].get("alternative", [])
                        if alternatives:
                            transcript = alternatives[0].get("transcript", "")
                            if transcript:
                                return True, transcript.strip()

        return False, "Could not decipher speech. Please speak clearly into the microphone."
    except Exception as e:
        return False, f"Speech recognition service error: {str(e)}"


def render_browser_voice_dictation_html(language_code: str = "en-IN") -> str:
    """
    Generates a modern, responsive HTML5 Web Speech API dictation widget.
    Allows real-time voice input directly into the browser in English, Hindi, and Marathi.
    """
    return f"""
    <div style="background: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <button id="micBtn" onclick="toggleSpeechRecognition()" style="
                    background: #2563EB; color: white; border: none; border-radius: 50%;
                    width: 44px; height: 44px; font-size: 1.2rem; cursor: pointer;
                    display: flex; align-items: center; justify-content: center;
                    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3); transition: all 0.2s;
                ">🎙️</button>
                <div>
                    <strong style="color: #F8FAFC; font-size: 0.95rem;">Voice Input / Speech-to-Text</strong><br>
                    <span id="speechStatus" style="color: #94A3B8; font-size: 0.82rem;">Click mic to speak in selected language</span>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <label style="color: #94A3B8; font-size: 0.85rem;">Language:</label>
                <select id="voiceLangSelect" onchange="updateLang()" style="
                    background: #0F172A; color: #F8FAFC; border: 1px solid #475569;
                    border-radius: 6px; padding: 0.3rem 0.6rem; font-size: 0.85rem;
                ">
                    <option value="en-IN" {"selected" if language_code == "en-IN" else ""}>English (India)</option>
                    <option value="hi-IN" {"selected" if language_code == "hi-IN" else ""}>Hindi (हिंदी)</option>
                    <option value="mr-IN" {"selected" if language_code == "mr-IN" else ""}>Marathi (मराठी)</option>
                </select>
            </div>
        </div>
        <div id="liveTranscript" style="
            display: none; margin-top: 0.8rem; padding: 0.6rem 0.8rem;
            background: #0F172A; border-radius: 8px; color: #38BDF8;
            font-size: 0.88rem; font-style: italic; border-left: 3px solid #38BDF8;
        "></div>
    </div>

    <script>
    let recognition = null;
    let isRecognizing = false;

    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {{
        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRec();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = document.getElementById('voiceLangSelect').value;

        recognition.onstart = function() {{
            isRecognizing = true;
            const btn = document.getElementById('micBtn');
            btn.style.background = '#EF4444';
            btn.innerHTML = '⏹️';
            document.getElementById('speechStatus').innerText = 'Listening... Speak now';
            document.getElementById('speechStatus').style.color = '#F87171';
            document.getElementById('liveTranscript').style.display = 'block';
        }};

        recognition.onresult = function(event) {{
            let interim = '';
            let final = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {{
                if (event.results[i].isFinal) {{
                    final += event.results[i][0].transcript + ' ';
                }} else {{
                    interim += event.results[i][0].transcript;
                }}
            }}
            document.getElementById('liveTranscript').innerText = 'Detected: ' + (final + interim);
            
            // Auto copy final text to clipboard with notification
            if (final) {{
                navigator.clipboard.writeText(final.trim());
                document.getElementById('speechStatus').innerText = 'Speech captured! (Copied to clipboard - paste into text area)';
                document.getElementById('speechStatus').style.color = '#34D399';
            }}
        }};

        recognition.onerror = function(event) {{
            document.getElementById('speechStatus').innerText = 'Voice error: ' + event.error;
            stopRecognition();
        }};

        recognition.onend = function() {{
            stopRecognition();
        }};
    }} else {{
        document.getElementById('speechStatus').innerText = 'Speech recognition not supported in this browser. Use Chrome/Edge.';
    }}

    function updateLang() {{
        if (recognition) {{
            recognition.lang = document.getElementById('voiceLangSelect').value;
        }}
    }}

    function toggleSpeechRecognition() {{
        if (!recognition) return;
        if (isRecognizing) {{
            recognition.stop();
        }} else {{
            recognition.lang = document.getElementById('voiceLangSelect').value;
            recognition.start();
        }}
    }}

    function stopRecognition() {{
        isRecognizing = false;
        const btn = document.getElementById('micBtn');
        if (btn) {{
            btn.style.background = '#2563EB';
            btn.innerHTML = '🎙️';
        }}
    }}
    </script>
    """


def render_tts_audio_player_html(text: str, language_code: str = "en-IN") -> str:
    """
    Renders browser Text-to-Speech audio synthesizer for listening to summaries.
    """
    clean_text = text.replace('"', '\\"').replace("'", "\\'").replace("\n", " ")
    return f"""
    <div style="margin-top: 0.8rem;">
        <button onclick="speakText('{clean_text}', '{language_code}')" style="
            background: #334155; color: #F8FAFC; border: 1px solid #475569;
            border-radius: 6px; padding: 0.4rem 0.9rem; font-size: 0.85rem;
            cursor: pointer; display: inline-flex; align-items: center; gap: 6px;
            transition: all 0.2s;
        ">
            🔊 Listen to Summary ({SUPPORTED_VOICE_LANGUAGES.get(language_code[:2], {}).get('name', 'Voice')})
        </button>
        <button onclick="window.speechSynthesis.cancel()" style="
            background: transparent; color: #94A3B8; border: 1px solid #334155;
            border-radius: 6px; padding: 0.4rem 0.6rem; font-size: 0.85rem;
            cursor: pointer; margin-left: 6px;
        ">
            ⏹️ Stop
        </button>
    </div>

    <script>
    function speakText(txt, lang) {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(txt);
            utterance.lang = lang;
            utterance.rate = 1.0;
            window.speechSynthesis.speak(utterance);
        }} else {{
            alert('Speech synthesis not supported in this browser.');
        }}
    }}
    </script>
    """
