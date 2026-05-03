"""
MultiCauseNet - Emotion-Cause Pair Extraction (ECPE)
Simple Streamlit demo focused on the core concept
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pipelines.text_pipeline import process_text_emotion, preprocess_text
from pipelines.audio_pipeline import process_audio_full
from pipelines.ecpe_module import extract_emotion_cause_pairs, format_ecpe_output
from utils.model_loader import preload_all_models

# Page config
st.set_page_config(
    page_title="MultiCauseNet - ECPE Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"  # Faster initial load
)

# Custom CSS - Simple and clean
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stAlert { margin: 1rem 0; }
    h1 { color: #667eea; text-align: center; }
    h2 { color: #764ba2; border-bottom: 2px solid #667eea; padding-bottom: 0.5rem; }
    .emotion-box {
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid;
        margin: 1rem 0;
    }
    .cause-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models (cached for fast reloads)
@st.cache_resource(show_spinner="Loading AI models... (first time only)")
def load_models():
    preload_all_models()

# Header
st.title("🧠 MultiCauseNet")
st.markdown("### Emotion-Cause Pair Extraction (ECPE)")
st.markdown("*Inspired by Dr. Hassan Nazeer's Research*")
st.divider()

# Load models (cached, only loads once)
load_models()

# Tabs for input
tab1, tab2 = st.tabs(["📝 Text Analysis", "🎤 Audio Analysis"])

# ===== TEXT ANALYSIS TAB =====
with tab1:
    st.markdown("## Text Input")
    
    # Example selector
    examples = {
        "Custom": "",
        "Joy - University Acceptance": "I am so happy today! I got accepted to my dream university and my parents are so proud of me. This is the best news ever!",
        "Sadness - Failed Exam": "I failed my final exam because I didn't study enough. I feel so disappointed in myself and worried about my future.",
        "Anger - Traffic": "The traffic made me late for the most important meeting of my career. I am so frustrated and angry at this situation!",
        "Fear - Strange Noises": "I heard strange noises in the basement. I'm really scared that something might be wrong. My heart is racing.",
        "Surprise - Party": "I can't believe it! My friends threw me a surprise party and got me the gift I've been wanting for years. I never expected this!"
    }
    
    selected = st.selectbox("Select an example or choose Custom:", list(examples.keys()))
    text_input = st.text_area(
        "Enter your text:",
        value=examples[selected],
        height=150,
        placeholder="Type or paste text here..."
    )
    
    if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
        if text_input.strip():
            with st.spinner("Analyzing..."):
                # Process
                processed_text = preprocess_text(text_input)
                emotion_result = process_text_emotion(processed_text)
                
                # ECPE for all detected emotions
                all_pairs = []
                for emotion_data in emotion_result.get('top_emotions', [emotion_result]):
                    pairs = extract_emotion_cause_pairs(
                        processed_text,
                        emotion_data['emotion'],
                        emotion_data['confidence']
                    )
                    all_pairs.extend(pairs)
                
                # Deduplicate and format
                seen_causes = set()
                unique_pairs = []
                for pair in sorted(all_pairs, key=lambda x: x['confidence'], reverse=True):
                    cause_key = pair['cause'].lower()
                    if cause_key not in seen_causes:
                        seen_causes.add(cause_key)
                        unique_pairs.append(pair)
                
                formatted_pairs = format_ecpe_output(unique_pairs[:5])
                
                # Display Results
                st.divider()
                st.markdown("## Results")
                
                # Multiple Emotions
                st.markdown("### Detected Emotions")
                top_emotions = emotion_result.get('top_emotions', [emotion_result])
                cols = st.columns(len(top_emotions))
                for i, (col, emotion) in enumerate(zip(cols, top_emotions)):
                    with col:
                        st.metric(
                            label=f"{'Primary' if i == 0 else f'Emotion #{i+1}'}",
                            value=emotion['emotion'].capitalize(),
                            delta=f"{emotion['confidence']*100:.1f}% confidence"
                        )
                
                # Emotion-Cause Pairs
                st.markdown("### 🔗 Emotion-Cause Pairs")
                
                if formatted_pairs:
                    for pair in formatted_pairs:
                        confidence_color = "🟢" if pair['confidence'] >= 0.75 else "🟡" if pair['confidence'] >= 0.5 else "🔴"
                        
                        with st.container():
                            st.markdown(f"""
                            <div class="cause-box">
                                <h4>Pair #{pair['pair_id']} {confidence_color} ({pair['confidence']*100:.0f}% confidence)</h4>
                                <p><strong>Emotion:</strong> {pair['emotion']}</p>
                                <p><strong>Cause:</strong> "{pair['cause']}"</p>
                                <p><em>Evidence: {pair['evidence']}</em></p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No clear emotion-cause pairs detected. Try a different text with more explicit causal language.")
        else:
            st.warning("Please enter some text to analyze.")

# ===== AUDIO ANALYSIS TAB =====
with tab2:
    st.markdown("## Audio Upload")
    
    audio_file = st.file_uploader(
        "Upload an audio file (WAV, MP3, etc.)",
        type=['wav', 'mp3', 'ogg', 'm4a', 'flac']
    )
    
    if st.button("🔍 Analyze Audio", type="primary", use_container_width=True):
        if audio_file:
            with st.spinner("Processing audio... (transcription + emotion analysis)"):
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name
                
                try:
                    # Process audio (now returns both text and audio emotions)
                    audio_result = process_audio_full(tmp_path)
                    
                    # Extract emotions
                    text_emotion = audio_result['text_emotion']
                    audio_emotion = audio_result['audio_emotion']
                    
                    # ECPE for all detected emotions
                    all_emotions = text_emotion.get('top_emotions', [text_emotion])
                    all_pairs = []
                    for emotion_data in all_emotions:
                        pairs = extract_emotion_cause_pairs(
                            audio_result['transcript'],
                            emotion_data['emotion'],
                            emotion_data['confidence']
                        )
                        all_pairs.extend(pairs)
                    
                    # Deduplicate and format
                    seen_causes = set()
                    unique_pairs = []
                    for pair in sorted(all_pairs, key=lambda x: x['confidence'], reverse=True):
                        cause_key = pair['cause'].lower()
                        if cause_key not in seen_causes:
                            seen_causes.add(cause_key)
                            unique_pairs.append(pair)
                    
                    formatted_pairs = format_ecpe_output(unique_pairs[:5])
                    
                    # Display Results
                    st.divider()
                    st.markdown("## Results")
                    
                    # Transcript
                    st.markdown("### Transcript")
                    st.info(audio_result['transcript'])
                    
                    # Analysis
                    if audio_result.get('analysis'):
                        st.markdown("### Analysis")
                        st.success(audio_result['analysis'])
                    
                    # Emotions - Text-based
                    st.markdown("### Text-Based Emotions (semantic content)")
                    text_emotions = text_emotion.get('top_emotions', [text_emotion])
                    cols = st.columns(len(text_emotions))
                    for i, (col, emotion) in enumerate(zip(cols, text_emotions)):
                        with col:
                            st.metric(
                                label=f"{'Primary' if i == 0 else f'Text #{i+1}'}",
                                value=emotion['emotion'].capitalize(),
                                delta=f"{emotion['confidence']*100:.1f}%"
                            )
                    
                    # Emotions - Audio-based
                    st.markdown("### Audio-Based Emotions (voice prosody)")
                    audio_emotions = audio_emotion.get('top_emotions', [audio_emotion])
                    cols = st.columns(len(audio_emotions))
                    for i, (col, emotion) in enumerate(zip(cols, audio_emotions)):
                        with col:
                            st.metric(
                                label=f"{'Primary' if i == 0 else f'Audio #{i+1}'}",
                                value=emotion['emotion'].capitalize(),
                                delta=f"{emotion['confidence']*100:.1f}%"
                            )
                    
                    # Emotion-Cause Pairs
                    st.markdown("### 🔗 Emotion-Cause Pairs")
                    
                    if formatted_pairs:
                        for pair in formatted_pairs:
                            confidence_color = "🟢" if pair['confidence'] >= 0.75 else "🟡" if pair['confidence'] >= 0.5 else "🔴"
                            
                            with st.container():
                                st.markdown(f"""
                                <div class="cause-box">
                                    <h4>Pair #{pair['pair_id']} {confidence_color} ({pair['confidence']*100:.0f}% confidence)</h4>
                                    <p><strong>Emotion:</strong> {pair['emotion']}</p>
                                    <p><strong>Cause:</strong> "{pair['cause']}"</p>
                                    <p><em>Evidence: {pair['evidence']}</em></p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No clear emotion-cause pairs detected.")
                
                finally:
                    # Cleanup
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        else:
            st.warning("Please upload an audio file.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>MultiCauseNet © 2024 | Inspired by Dr. Hassan Nazeer's ECPE Research</p>
</div>
""", unsafe_allow_html=True)
