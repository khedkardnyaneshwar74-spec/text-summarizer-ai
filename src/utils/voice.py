"""
Voice & Audio Summarization module supporting English, Hindi (हिंदी), and Marathi (मराठी).
Provides real-time browser speech recognition, local Whisper speech-to-text, and Text-to-Speech playback.
"""

import io
import os
import tempfile
from typing import Tuple, Optional
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal
import torch

# Global cached Whisper model and processor
_WHISPER_PROCESSOR = None
_WHISPER_MODEL = None

SUPPORTED_VOICE_LANGUAGES = {
    "en": {"code": "en-IN", "whisper_lang": "english", "name": "English"},
    "hi": {"code": "hi-IN", "whisper_lang": "hindi", "name": "Hindi (हिंदी)"},
    "mr": {"code": "mr-IN", "whisper_lang": "marathi", "name": "Marathi (मराठी)"},
}


def get_whisper_components():
    """
    Load or retrieve singleton Whisper model and processor for local offline transcription.
    Operates without external ffmpeg dependencies by utilizing native scipy WAV decoding.
    """
    global _WHISPER_PROCESSOR, _WHISPER_MODEL
    if _WHISPER_PROCESSOR is not None and _WHISPER_MODEL is not None:
        return _WHISPER_PROCESSOR, _WHISPER_MODEL

    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model.to(device)
        model.eval()

        _WHISPER_PROCESSOR = processor
        _WHISPER_MODEL = model
        return _WHISPER_PROCESSOR, _WHISPER_MODEL
    except Exception as e:
        return None, None


def transcribe_audio_bytes(audio_bytes: bytes, language_code: str = "en") -> Tuple[bool, str]:
    """
    Transcribe audio bytes (from st.audio_input or file upload) using local Whisper model.
    Decodes WAV natively via scipy.io.wavfile to bypass ffmpeg binary requirements on Windows.
    Supports English, Hindi, and Marathi with 0 external API cost or network dependencies.
    """
    if not audio_bytes or len(audio_bytes) < 100:
        return False, "Audio recording is empty or too short."

    # Map language codes for Whisper
    lang_key = language_code[:2].lower()
    target_lang = SUPPORTED_VOICE_LANGUAGES.get(lang_key, {}).get("whisper_lang", "english")

    try:
        # 1. Native WAV decoding via scipy
        try:
            samplerate, audio_data = wavfile.read(io.BytesIO(audio_bytes))
        except Exception as wav_err:
            return False, f"Could not read audio format as WAV: {str(wav_err)}. Please upload a valid .wav audio file."

        # 2. Stereo to mono conversion
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # 3. Resample to 16,000 Hz if needed (Whisper requirement)
        if samplerate != 16000 and len(audio_data) > 0:
            target_samples = int(len(audio_data) * 16000 / samplerate)
            audio_data = scipy.signal.resample(audio_data, target_samples)
            samplerate = 16000

        # 4. Normalize to float32 between -1.0 and 1.0
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_float = audio_data.astype(np.float32) / 2147483648.0
        elif np.issubdtype(audio_data.dtype, np.floating):
            audio_float = audio_data.astype(np.float32)
            # Clip if max amplitude exceeds 1.0
            max_val = np.max(np.abs(audio_float))
            if max_val > 1.0:
                audio_float = audio_float / max_val
        else:
            audio_float = audio_data.astype(np.float32) / 32768.0

        if len(audio_float) < 1600:  # Less than 0.1s
            return False, "Audio recording too short to contain recognizable speech."

        # 5. Whisper inference
        processor, model = get_whisper_components()
        if processor is None or model is None:
            return False, "Speech recognition model could not be loaded."

        device = next(model.parameters()).device
        inputs = processor(audio_float, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=target_lang, task="transcribe")
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=448,
            )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        if transcription:
            return True, transcription
        return False, "No audible speech recognized in the audio recording."

    except Exception as e:
        return False, f"Audio transcription error: {str(e)}"


# Alias for backward compatibility
transcribe_wav_audio_bytes = transcribe_audio_bytes


def render_browser_voice_dictation_html(language_code: str = "en-IN") -> str:
    """
    Generates a modern, responsive HTML5 Web Speech API dictation widget with gradient aesthetics.
    Provides direct URL parameter transfer and automatic clipboard backup.
    """
    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 0 15px rgba(99, 102, 241, 0.15);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <button id="micBtn" onclick="toggleSpeechRecognition()" title="Click to start/stop dictation" style="
                    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
                    color: white; border: none; border-radius: 50%;
                    width: 52px; height: 52px; font-size: 1.4rem; cursor: pointer;
                    display: flex; align-items: center; justify-content: center;
                    box-shadow: 0 4px 18px rgba(124, 58, 237, 0.45);
                    transition: transform 0.2s, box-shadow 0.2s, background 0.2s;
                ">🎙️</button>
                <div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <strong style="color: #F8FAFC; font-size: 1.05rem; letter-spacing: -0.01em;">Live Browser Voice Dictation</strong>
                        <span style="background: rgba(99, 102, 241, 0.25); color: #A5B4FC; font-size: 0.72rem; padding: 2px 8px; border-radius: 9999px; font-weight: 600; border: 1px solid rgba(99, 102, 241, 0.4);">Web Speech API</span>
                    </div>
                    <span id="speechStatus" style="color: #94A3B8; font-size: 0.84rem;">Click microphone to dictate in English, Hindi, or Marathi</span>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                <label style="color: #CBD5E1; font-size: 0.85rem; font-weight: 500;">Spoken Language:</label>
                <select id="voiceLangSelect" onchange="updateLang()" style="
                    background: #0F172A; color: #F8FAFC; border: 1px solid #6366F1;
                    border-radius: 8px; padding: 0.45rem 0.85rem; font-size: 0.85rem;
                    outline: none; cursor: pointer;
                ">
                    <option value="en-IN" {"selected" if language_code == "en-IN" else ""}>English (India)</option>
                    <option value="hi-IN" {"selected" if language_code == "hi-IN" else ""}>Hindi (हिंदी)</option>
                    <option value="mr-IN" {"selected" if language_code == "mr-IN" else ""}>Marathi (मराठी)</option>
                </select>
                <button onclick="applyTranscriptDirectly()" id="applyBtn" style="
                    display: none; background: linear-gradient(135deg, #10B981 0%, #059669 100%);
                    color: white; border: none; border-radius: 8px; padding: 0.45rem 1rem;
                    font-size: 0.85rem; cursor: pointer; font-weight: 600;
                    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.35);
                ">📥 Insert to Source Text</button>
            </div>
        </div>
        <div id="liveTranscript" style="
            display: none; margin-top: 1rem; padding: 0.85rem 1.1rem;
            background: rgba(15, 23, 42, 0.95); border-radius: 10px; color: #38BDF8;
            font-size: 0.94rem; border-left: 4px solid #38BDF8; line-height: 1.55;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
        "></div>
    </div>

    <script>
    let recognition = null;
    let isRecognizing = false;
    let accumulatedTranscript = '';

    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {{
        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRec();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = document.getElementById('voiceLangSelect').value;

        recognition.onstart = function() {{
            isRecognizing = true;
            accumulatedTranscript = '';
            const btn = document.getElementById('micBtn');
            btn.style.background = 'linear-gradient(135deg, #EF4444 0%, #DC2626 100%)';
            btn.innerHTML = '⏹️';
            btn.style.boxShadow = '0 0 20px rgba(239, 68, 68, 0.6)';
            document.getElementById('speechStatus').innerText = '🔴 Listening actively... Speak now in your selected language';
            document.getElementById('speechStatus').style.color = '#F87171';
            document.getElementById('liveTranscript').style.display = 'block';
            document.getElementById('liveTranscript').innerText = 'Awaiting spoken input...';
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

            if (final) {{
                accumulatedTranscript += final;
            }}

            const currentDisplay = (accumulatedTranscript + interim).trim();
            document.getElementById('liveTranscript').innerText = currentDisplay;

            if (accumulatedTranscript.trim()) {{
                document.getElementById('applyBtn').style.display = 'inline-block';
            }}
        }};

        recognition.onerror = function(event) {{
            document.getElementById('speechStatus').innerText = 'Microphone notice: ' + (event.error || 'No speech heard.');
            stopRecognition();
        }};

        recognition.onend = function() {{
            stopRecognition();
            if (accumulatedTranscript.trim()) {{
                document.getElementById('speechStatus').innerText = '✅ Speech captured! Auto-sending directly into document...';
                document.getElementById('speechStatus').style.color = '#34D399';
                applyTranscriptDirectly();
            }}
        }};
    }} else {{
        document.getElementById('speechStatus').innerText = 'Real-time Web Speech API supported in Chrome, Edge, and Safari.';
    }}

    function applyTranscriptDirectly() {{
        const textToInject = (accumulatedTranscript || document.getElementById('liveTranscript').innerText).trim();
        if (!textToInject) return;

        // Copy to clipboard as immediate backup
        try {{
            navigator.clipboard.writeText(textToInject);
        }} catch(e) {{}}

        // Send via URL parameter to Streamlit
        try {{
            const pUrl = new URL(window.parent.location.href);
            pUrl.searchParams.set('spoken_doc', encodeURIComponent(textToInject));
            window.parent.location.href = pUrl.href;
            document.getElementById('speechStatus').innerText = '✅ Direct transfer complete! Updating document...';
        }} catch (err) {{
            // Direct DOM fallback
            try {{
                const parentTextareas = window.parent.document.querySelectorAll('textarea');
                for (let ta of parentTextareas) {{
                    if (ta.placeholder && (ta.placeholder.includes('Paste') || ta.placeholder.includes('Source'))) {{
                        ta.value = textToInject;
                        ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        ta.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        document.getElementById('speechStatus').innerText = '✅ Injected directly into document textarea!';
                        return;
                    }}
                }}
            }} catch(e2) {{}}
            document.getElementById('speechStatus').innerText = '✅ Speech copied to clipboard! Paste directly into text box.';
        }}
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
            btn.style.background = 'linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%)';
            btn.innerHTML = '🎙️';
            btn.style.boxShadow = '0 4px 18px rgba(124, 58, 237, 0.45)';
        }}
    }}
    </script>
    """


def render_tts_audio_player_html(text: str, language_code: str = "en-IN") -> str:
    """
    Renders browser Text-to-Speech audio synthesizer for listening to summaries with gradient styling.
    """
    clean_text = text.replace('"', '\\"').replace("'", "\\'").replace("\n", " ").replace("\r", "")
    lang_name = "English" if "en" in language_code else ("Hindi (हिंदी)" if "hi" in language_code else "Marathi (मराठी)")
    return f"""
    <div style="
        margin-top: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.75rem 1rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
    ">
        <button onclick="speakText('{clean_text}', '{language_code}')" style="
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
            color: #FFFFFF; border: none; border-radius: 8px;
            padding: 0.5rem 1.1rem; font-size: 0.88rem; cursor: pointer;
            display: inline-flex; align-items: center; gap: 8px; font-weight: 600;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35); transition: transform 0.15s;
        ">
            🔊 Listen to Summary ({lang_name})
        </button>
        <button onclick="window.speechSynthesis.cancel()" style="
            background: rgba(51, 65, 85, 0.8); color: #E2E8F0; border: 1px solid #475569;
            border-radius: 8px; padding: 0.5rem 0.9rem; font-size: 0.88rem;
            cursor: pointer; font-weight: 500;
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
        }}
    }}
    </script>
    """
