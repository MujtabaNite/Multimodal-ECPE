"""
Model Loader - Singleton pattern for loading and caching models
Ensures models are loaded only once and reused across requests
"""

import os
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    pipeline
)
try:
    from transformers import HubertForSequenceClassification
    HUBERT_AVAILABLE = True
except ImportError:
    HUBERT_AVAILABLE = False
import whisper
from pathlib import Path

# Global cache for models
_MODEL_CACHE = {}

def get_device():
    """Get the best available device (GPU if available, else CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_text_emotion_model():
    """
    Load text emotion classification model
    Using: j-hartmann/emotion-english-distilroberta-base (Enhanced model, 7 emotions)
    Better accuracy than DistilBERT base
    Cached to avoid reloading
    """
    if 'text_emotion' not in _MODEL_CACHE:
        print("Loading enhanced text emotion model...")
        # Using better emotion model with RoBERTa
        model_name = "j-hartmann/emotion-english-distilroberta-base"
        device = get_device()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir="./models/text_emotion"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir="./models/text_emotion"
        ).to(device)
        
        _MODEL_CACHE['text_emotion'] = {
            'model': model,
            'tokenizer': tokenizer,
            'device': device
        }
        print(f"Enhanced text emotion model loaded on {device}")
    
    return _MODEL_CACHE['text_emotion']

def get_audio_emotion_model():
    """
    Load BEST audio emotion recognition model
    Using: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (State-of-the-art)
    This is the BEST available model - large, robust, fine-tuned on emotion dataset
    Falls back to HuBERT-base, then Wav2Vec2 if unavailable
    Cached to avoid reloading
    """
    if 'audio_emotion' not in _MODEL_CACHE:
        print("Loading BEST audio emotion model (wav2vec2-large-robust, emotion fine-tuned)...")
        device = get_device()
        
        # Try best model first: audeering wav2vec2-large-robust fine-tuned for emotion
        # This is state-of-the-art for emotion recognition
        model_loaded = False
        
        try:
            model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
            print(f"Attempting to load: {model_name}")
            
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_name,
                cache_dir="./models/audio_emotion"
            )
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_name,
                cache_dir="./models/audio_emotion"
            ).to(device)
            
            print(f"✓ BEST audio emotion model loaded (wav2vec2-large-robust) on {device}")
            model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load audeering model: {e}")
            print("Trying HuBERT-base (SUPERB) as fallback...")
            
            # Fallback 1: HuBERT-base SUPERB
            if HUBERT_AVAILABLE:
                try:
                    model_name = "superb/hubert-base-superb-er"
                    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                        model_name,
                        cache_dir="./models/audio_emotion"
                    )
                    model = HubertForSequenceClassification.from_pretrained(
                        model_name,
                        cache_dir="./models/audio_emotion"
                    ).to(device)
                    print(f"✓ Audio emotion model (HuBERT-base, SUPERB) loaded on {device}")
                    model_loaded = True
                except Exception as e2:
                    print(f"Warning: Could not load HuBERT: {e2}")
            
            # Fallback 2: Original Wav2Vec2
            if not model_loaded:
                try:
                    print("Falling back to Wav2Vec2 model...")
                    model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                        model_name,
                        cache_dir="./models/audio_emotion"
                    )
                    model = Wav2Vec2ForSequenceClassification.from_pretrained(
                        model_name,
                        cache_dir="./models/audio_emotion"
                    ).to(device)
                    print(f"✓ Audio emotion model (Wav2Vec2) loaded on {device}")
                    model_loaded = True
                except Exception as e3:
                    print(f"ERROR: All models failed to load: {e3}")
                    raise
        
        _MODEL_CACHE['audio_emotion'] = {
            'model': model,
            'feature_extractor': feature_extractor,
            'device': device
        }
    
    return _MODEL_CACHE['audio_emotion']

def get_whisper_model():
    """
    Load Whisper ASR model for offline speech-to-text
    Using: tiny model for FAST loading and processing
    Cached to avoid reloading
    """
    if 'whisper' not in _MODEL_CACHE:
        print("Loading Whisper ASR model (tiny - optimized for speed)...")
        # Use 'tiny' model - fastest option, good enough for demo
        # tiny: 39M params, ~32MB, ~10x faster than base
        # base: 74M params, ~145MB
        model = whisper.load_model("tiny", download_root="./models/whisper")
        _MODEL_CACHE['whisper'] = model
        print("Whisper ASR model loaded (tiny)")
    
    return _MODEL_CACHE['whisper']

def get_spacy_model():
    """
    Load spaCy model for NLP processing (dependency parsing, NER)
    Used in ECPE module for cause extraction
    """
    if 'spacy' not in _MODEL_CACHE:
        print("Loading spaCy model...")
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
        
        _MODEL_CACHE['spacy'] = nlp
        print("spaCy model loaded")
    
    return _MODEL_CACHE['spacy']

def preload_all_models():
    """
    Preload all models at startup to avoid latency during demo
    Call this when the Streamlit app starts
    """
    print("\n" + "="*60)
    print("PRELOADING ALL MODELS FOR MULTICAUSENET DEMO")
    print("="*60 + "\n")
    
    get_text_emotion_model()
    get_audio_emotion_model()
    get_whisper_model()
    get_spacy_model()
    
    print("\n" + "="*60)
    print("ALL MODELS LOADED AND CACHED")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Test model loading
    preload_all_models()
    print("Model loader test complete!")

