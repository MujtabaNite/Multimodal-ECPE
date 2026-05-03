"""
MultiCauseNet - Emotion-Cause Pair Extraction (ECPE)
Professional Flask application focused on core ECPE functionality
"""

from flask import Flask, request, jsonify, render_template
import sys
from pathlib import Path
import tempfile
import os
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pipelines.text_pipeline import process_text_emotion, preprocess_text
from pipelines.audio_pipeline import process_audio_full
from pipelines.ecpe_module import extract_emotion_cause_pairs, format_ecpe_output
from utils.model_loader import preload_all_models

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates_flask',
            static_folder='static_flask')

# Configure file upload
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Optimize Flask performance
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # Cache static files for 1 year
app.config['TEMPLATES_AUTO_RELOAD'] = False  # Disable in production

# Load models on startup (in background to not block)
print("\n" + "="*60)
print("MultiCauseNet - Loading AI Models (Optimized)")
print("="*60)

import threading
model_loading_complete = threading.Event()

def load_models_background():
    preload_all_models()
    model_loading_complete.set()
    print("="*60)
    print("All models loaded and ready!")
    print("="*60 + "\n")

# Start loading in background
loading_thread = threading.Thread(target=load_models_background, daemon=True)
loading_thread.start()

print("Models loading in background...")
print("Server will be ready shortly!")
print("="*60 + "\n")

@app.route('/')
def index():
    """Main page"""
    response = render_template('index.html')
    return response

@app.route('/api/status', methods=['GET'])
def status():
    """Check if models are loaded"""
    return jsonify({
        'models_ready': model_loading_complete.is_set(),
        'message': 'Models ready' if model_loading_complete.is_set() else 'Models loading...'
    })

@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    """Analyze text for emotions and causes"""
    # Wait for models to be ready
    if not model_loading_complete.is_set():
        return jsonify({'error': 'Models are still loading, please wait...'}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        
        if not text or not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Preprocess
        text = preprocess_text(text)
        
        # Emotion detection (now returns top 3 emotions)
        emotion_result = process_text_emotion(text)
        
        # ECPE for all detected emotions
        all_pairs = []
        for emotion_data in emotion_result.get('top_emotions', [emotion_result]):
            pairs = extract_emotion_cause_pairs(
                text,
                emotion_data['emotion'],
                emotion_data['confidence']
            )
            all_pairs.extend(pairs)
        
        # Sort by confidence and remove duplicates
        seen_causes = set()
        unique_pairs = []
        for pair in sorted(all_pairs, key=lambda x: x['confidence'], reverse=True):
            cause_key = pair['cause'].lower()
            if cause_key not in seen_causes:
                seen_causes.add(cause_key)
                unique_pairs.append(pair)
        
        formatted_pairs = format_ecpe_output(unique_pairs[:5])  # Top 5 pairs
        
        return jsonify({
            'success': True,
            'emotions': emotion_result.get('top_emotions', [emotion_result]),  # Multiple emotions
            'causes': formatted_pairs,
            'input_text': text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/audio', methods=['POST'])
def analyze_audio():
    """Analyze audio for emotions and causes"""
    # Wait for models to be ready
    if not model_loading_complete.is_set():
        return jsonify({'error': 'Models are still loading, please wait...'}), 503
    
    try:
        print("DEBUG: Received audio upload request")
        print(f"DEBUG: Files in request: {list(request.files.keys())}")
        
        if 'audio' not in request.files:
            print("DEBUG: No 'audio' key in request.files")
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        print(f"DEBUG: Audio filename: {audio_file.filename}")
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Determine file extension
        file_ext = os.path.splitext(audio_file.filename)[1].lower()
        if not file_ext:
            file_ext = '.wav'
        
        # Save temporarily with correct extension
        # Use proper file handling to ensure file is saved correctly
        import shutil
        tmp_dir = tempfile.gettempdir()
        tmp_filename = f"multicausenet_audio_{os.getpid()}_{int(time.time())}{file_ext}"
        tmp_path = os.path.join(tmp_dir, tmp_filename)
        
        # Save the file
        audio_file.save(tmp_path)
        print(f"DEBUG: Saved audio to: {tmp_path}")
        print(f"DEBUG: File size: {os.path.getsize(tmp_path)} bytes")
        
        # Verify file exists and is not empty
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            return jsonify({'error': 'File upload failed or file is empty'}), 400
        
        try:
            print("DEBUG: Processing audio file...")
            # Process audio (now includes text emotion analysis)
            audio_result = process_audio_full(tmp_path)
            print(f"DEBUG: Transcript: {audio_result.get('transcript', 'N/A')}")
            
            # Extract emotions from result
            text_emotion = audio_result['text_emotion']
            audio_emotion = audio_result['audio_emotion']
            
            # ECPE for all detected emotions (combine text and audio)
            all_emotions = text_emotion.get('top_emotions', [text_emotion])
            all_pairs = []
            
            for emotion_data in all_emotions:
                pairs = extract_emotion_cause_pairs(
                    audio_result['transcript'],
                    emotion_data['emotion'],
                    emotion_data['confidence']
                )
                all_pairs.extend(pairs)
            
            # Sort by confidence and remove duplicates
            seen_causes = set()
            unique_pairs = []
            for pair in sorted(all_pairs, key=lambda x: x['confidence'], reverse=True):
                cause_key = pair['cause'].lower()
                if cause_key not in seen_causes:
                    seen_causes.add(cause_key)
                    unique_pairs.append(pair)
            
            formatted_pairs = format_ecpe_output(unique_pairs[:5])
            
            return jsonify({
                'success': True,
                'transcript': audio_result['transcript'],
                'language': audio_result['language'],
                'text_emotions': text_emotion.get('top_emotions', [text_emotion]),
                'audio_emotions': audio_emotion.get('top_emotions', [audio_emotion]),
                'analysis': audio_result.get('analysis', ''),
                'causes': formatted_pairs
            })
        
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                print("DEBUG: Cleaned up temp file")
    
    except Exception as e:
        print(f"DEBUG: Error in analyze_audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
