"""
Audio Processing Pipeline
Handles audio input: ASR (speech-to-text) + audio emotion recognition
"""

import torch
import librosa
import numpy as np
import soundfile as sf
from utils.model_loader import get_whisper_model, get_audio_emotion_model

def transcribe_audio(audio_path):
    """
    Perform automatic speech recognition using Whisper
    
    Args:
        audio_path (str): Path to audio file
        
    Returns:
        dict: {
            'transcript': str,
            'language': str,
            'segments': list
        }
    """
    model = get_whisper_model()
    
    # Load audio using librosa (avoids FFmpeg dependency issue)
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Transcribe audio - pass numpy array instead of file path
    result = model.transcribe(
        audio,
        fp16=False,  # Use fp32 for CPU compatibility
        language='en'  # Can be set to None for auto-detection
    )
    
    return {
        'transcript': result['text'].strip(),
        'language': result.get('language', 'en'),
        'segments': result.get('segments', [])
    }

def recognize_audio_emotion(audio_path, sample_rate=16000):
    """
    Recognize emotion from audio signal
    Uses enhanced HuBERT model for better accuracy
    
    Args:
        audio_path (str): Path to audio file
        sample_rate (int): Target sample rate (16kHz for HuBERT/Wav2Vec2)
        
    Returns:
        dict: {
            'emotion': str,
            'confidence': float,
            'all_emotions': dict,
            'top_emotions': list
        }
    """
    # Get cached model
    model_dict = get_audio_emotion_model()
    model = model_dict['model']
    feature_extractor = model_dict['feature_extractor']
    device = model_dict['device']
    
    # Load and resample audio (16kHz for models)
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        print(f"Error loading audio: {e}")
        # Try alternative loading
        import soundfile as sf
        audio, sr = sf.read(audio_path)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    
    # Ensure audio is in correct format (mono, float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    audio = audio.astype(np.float32)
    
    # Normalize audio if too quiet or too loud
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.95  # Normalize to prevent clipping
    
    # Prepare input features
    inputs = feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get predictions
    predicted_class_id = probabilities.argmax().item()
    confidence = probabilities[0][predicted_class_id].item()
    
    # Emotion labels - handle both HuBERT (hap/neu/ang/etc) and standard labels
    if hasattr(model.config, 'id2label') and model.config.id2label:
        id2label = model.config.id2label
        raw_label = id2label[predicted_class_id]
    else:
        # Fallback label mapping for HuBERT SUPERB model
        emotion_labels = ['hap', 'neu', 'ang', 'sad', 'fea', 'sur', 'dis']
        raw_label = emotion_labels[predicted_class_id] if predicted_class_id < len(emotion_labels) else 'neu'
        id2label = {i: emotion_labels[i] for i in range(len(emotion_labels))}
    
    # Map IEMOCAP/HuBERT labels to standard labels
    label_mapping = {
        'hap': 'joy',
        'happy': 'joy',
        'neu': 'neutral',
        'neutral': 'neutral',
        'ang': 'anger',
        'angry': 'anger',
        'sad': 'sadness',
        'sadness': 'sadness',
        'fea': 'fear',
        'fear': 'fear',
        'sur': 'surprise',
        'surprise': 'surprise',
        'dis': 'mixEmotions',
        'disgust': 'disgust'
    }
    
    predicted_emotion = label_mapping.get(raw_label.lower(), raw_label.lower())
    
    # Get all emotion probabilities with mapping
    all_emotions = {}
    for i in range(probabilities.shape[1]):
        if hasattr(model.config, 'id2label') and i in id2label:
            raw_emotion = id2label[i]
        else:
            emotion_labels = ['hap', 'neu', 'ang', 'sad', 'fea', 'sur', 'dis']
            raw_emotion = emotion_labels[i] if i < len(emotion_labels) else f'emotion_{i}'
        
        # Map to standard label
        standard_label = label_mapping.get(raw_emotion.lower(), raw_emotion.lower())
        all_emotions[standard_label] = probabilities[0][i].item()
    
    # Get top 3 emotions
    sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
    top_emotions = [
        {'emotion': emotion, 'confidence': float(conf)}
        for emotion, conf in sorted_emotions[:3]
        if conf > 0.1  # Only include emotions with >10% confidence
    ]
    
    return {
        'emotion': predicted_emotion,
        'confidence': float(confidence),
        'all_emotions': all_emotions,
        'top_emotions': top_emotions  # Top 3 emotions with scores
    }

def process_audio_full(audio_path):
    """
    Complete audio pipeline: ASR + emotion recognition + comparison
    
    Args:
        audio_path (str): Path to audio file
        
    Returns:
        dict: {
            'transcript': str,
            'text_length': int,
            'audio_emotion': dict,
            'language': str,
            'analysis': str (explanation of differences)
        }
    """
    # Step 1: Transcribe audio
    asr_result = transcribe_audio(audio_path)
    transcript = asr_result['transcript']
    
    # Step 2: Recognize emotion from audio signal (prosody, tone, pitch)
    audio_emotion = recognize_audio_emotion(audio_path)
    
    # Step 3: Analyze text emotion (semantic content)
    from pipelines.text_pipeline import process_text_emotion
    text_emotion = process_text_emotion(transcript)
    
    # Step 4: Compare and explain
    analysis = _compare_emotions(text_emotion, audio_emotion)
    
    return {
        'transcript': transcript,
        'text_length': len(transcript),
        'audio_emotion': audio_emotion,  # From voice prosody
        'text_emotion': text_emotion,     # From semantic content
        'language': asr_result['language'],
        'analysis': analysis
    }

def _compare_emotions(text_emotion, audio_emotion):
    """
    Compare text and audio emotions and provide explanation
    """
    text_top = text_emotion['emotion']
    audio_top = audio_emotion['emotion']
    
    if text_top.lower() == audio_top.lower():
        return f"Text and audio emotions align - both indicate {text_top}. This suggests genuine emotion expression."
    else:
        return (f"Text emotion ({text_top}) differs from audio emotion ({audio_top}). "
                f"This can indicate: (1) Mixed emotions, (2) Sarcasm/irony, "
                f"(3) Emotional masking, or (4) Complex emotional state.")


if __name__ == "__main__":
    # Test the pipeline
    import sys
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(f"Testing Audio Pipeline with: {audio_file}\n")
        result = process_audio_full(audio_file)
        print(f"Transcript: {result['transcript']}")
        print(f"Language: {result['language']}")
        print(f"Audio Emotion: {result['audio_emotion']['emotion']} "
              f"(confidence: {result['audio_emotion']['confidence']:.3f})")
    else:
        print("Usage: python audio_pipeline.py <audio_file>")

