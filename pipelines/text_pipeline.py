"""
Text Emotion Classification Pipeline
Processes text input and returns emotion predictions with confidence scores
"""

import torch
import torch.nn.functional as F
from utils.model_loader import get_text_emotion_model

def process_text_emotion(text):
    """
    Process text input and predict emotion with confidence
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        dict: {
            'emotion': str,
            'confidence': float,
            'all_emotions': dict (emotion -> confidence)
        }
    """
    if not text or not text.strip():
        return {
            'emotion': 'neutral',
            'confidence': 1.0,
            'all_emotions': {'neutral': 1.0}
        }
    
    # Get cached model
    model_dict = get_text_emotion_model()
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    device = model_dict['device']
    
    # Tokenize and prepare input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
    
    # Get predictions
    predicted_class_id = probabilities.argmax().item()
    confidence = probabilities[0][predicted_class_id].item()
    
    # Emotion labels from the model
    id2label = model.config.id2label
    predicted_emotion = id2label[predicted_class_id]
    
    # Get all emotion probabilities
    all_emotions = {
        id2label[i]: probabilities[0][i].item()
        for i in range(len(id2label))
    }
    
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

def preprocess_text(text):
    """
    Preprocess text input (cleaning, normalization)
    """
    if not text:
        return ""
    
    # Basic cleaning
    text = text.strip()
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    return text

if __name__ == "__main__":
    # Test the pipeline
    test_texts = [
        "I am so happy today! Everything is going great!",
        "I feel really sad and disappointed.",
        "This makes me so angry!",
        "I'm afraid something bad will happen."
    ]
    
    print("Testing Text Emotion Pipeline:\n")
    for text in test_texts:
        result = process_text_emotion(text)
        print(f"Text: {text}")
        print(f"Emotion: {result['emotion']} (confidence: {result['confidence']:.3f})")
        print(f"All emotions: {result['all_emotions']}\n")

