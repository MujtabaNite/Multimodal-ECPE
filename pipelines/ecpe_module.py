"""
Emotion-Cause Pair Extraction (ECPE) Module
Offline, rule-based approach using dependency parsing and heuristics
No cloud LLMs - fully deterministic and explainable

Inspired by Dr. Hassan Nazeer's research on explainable emotion-cause analysis
"""

import re
from utils.model_loader import get_spacy_model
import numpy as np

# Emotion trigger words for cause detection
EMOTION_TRIGGERS = {
    'joy': ['happy', 'glad', 'excited', 'delighted', 'pleased', 'joyful', 'wonderful', 'great', 'amazing', 'fantastic'],
    'sadness': ['sad', 'unhappy', 'disappointed', 'depressed', 'miserable', 'upset', 'hurt', 'heartbroken', 'gloomy'],
    'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated', 'outraged', 'enraged'],
    'fear': ['afraid', 'scared', 'frightened', 'terrified', 'anxious', 'worried', 'nervous', 'fearful'],
    'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'startled', 'unexpected'],
    'disgust': ['disgusted', 'revolted', 'repulsed', 'sick', 'nauseated'],
    'neutral': []
}

# Enhanced causal connectives and markers
CAUSAL_MARKERS = [
    # Direct causation
    'because', 'since', 'as', 'due to', 'owing to', 'caused by',
    'because of', 'thanks to', 'on account of', 'as a result of',
    'given that', 'considering', 'seeing that', 'in light of',
    
    # Consequence markers
    'therefore', 'thus', 'so', 'hence', 'consequently', 'accordingly',
    'as a consequence', 'for this reason', 'that is why',
    
    # Action/event triggers
    'leads to', 'results in', 'makes me', 'made me', 'makes', 'made',
    'causes', 'caused', 'triggers', 'triggered',
    
    # Temporal/contextual
    'when', 'after', 'upon', 'following', 'ever since', 'now that',
    
    # Perception/cognition verbs
    'seeing', 'hearing', 'knowing', 'learning', 'finding', 'realizing',
    'discovering', 'noticing', 'observing', 'understanding'
]

def split_into_sentences(text):
    """Split text into sentences using spaCy (optimized)"""
    nlp = get_spacy_model()
    # Disable unnecessary pipeline components for speed
    doc = nlp(text, disable=['ner', 'textcat'])
    return [sent.text.strip() for sent in doc.sents]

def extract_clauses(sentence):
    """
    Extract clauses from a sentence using dependency parsing
    Returns list of clause dictionaries
    """
    nlp = get_spacy_model()
    doc = nlp(sentence)
    
    clauses = []
    
    # Find main verb and its dependents
    for token in doc:
        if token.pos_ == 'VERB':
            # Get the clause around this verb
            clause_tokens = [token]
            
            # Get subjects
            for child in token.children:
                if child.dep_ in ['nsubj', 'nsubjpass']:
                    clause_tokens.extend([child] + list(child.subtree))
            
            # Get objects
            for child in token.children:
                if child.dep_ in ['dobj', 'iobj', 'pobj', 'attr']:
                    clause_tokens.extend([child] + list(child.subtree))
            
            # Get adverbial modifiers
            for child in token.children:
                if child.dep_ in ['advmod', 'prep', 'advcl']:
                    clause_tokens.extend([child] + list(child.subtree))
            
            clause_tokens = sorted(set(clause_tokens), key=lambda t: t.i)
            clause_text = ' '.join([t.text for t in clause_tokens])
            
            if len(clause_text.split()) >= 2:  # At least 2 words
                clauses.append({
                    'text': clause_text,
                    'root_verb': token.text,
                    'pos': token.idx
                })
    
    return clauses if clauses else [{'text': sentence, 'root_verb': None, 'pos': 0}]

def detect_causal_relationship(sentence):
    """
    Detect if sentence contains explicit causal markers
    Returns (has_marker, marker_type, split_point)
    """
    sentence_lower = sentence.lower()
    
    for marker in CAUSAL_MARKERS:
        if marker in sentence_lower:
            return True, marker, sentence_lower.find(marker)
    
    return False, None, -1

def emotion_keyword_in_text(text, emotion):
    """
    Check if emotion-related keywords appear in text
    """
    text_lower = text.lower()
    emotion_lower = emotion.lower()
    
    # Map various emotion labels to standard categories
    emotion_mapping = {
        'joy': ['joy', 'happiness', 'happy'],
        'sadness': ['sadness', 'sad', 'sorrow'],
        'anger': ['anger', 'angry', 'mad'],
        'fear': ['fear', 'afraid', 'scared'],
        'surprise': ['surprise', 'surprised'],
        'disgust': ['disgust', 'disgusted'],
        'love': ['love', 'joy'],  # Map love to joy
        'neutral': ['neutral']
    }
    
    # Find matching category
    for category, keywords in emotion_mapping.items():
        if emotion_lower in keywords:
            triggers = EMOTION_TRIGGERS.get(category, [])
            for trigger in triggers:
                if trigger in text_lower:
                    return True, trigger
            break
    
    return False, None

def extract_emotion_cause_pairs(text, emotion, confidence):
    """
    Enhanced ECPE function with multiple sophisticated strategies
    
    Args:
        text (str): Input text
        emotion (str): Predicted emotion label
        confidence (float): Emotion confidence score
        
    Returns:
        list: List of emotion-cause pair dictionaries
    """
    if not text or not text.strip():
        return []
    
    sentences = split_into_sentences(text)
    pairs = []
    
    # Strategy 1: Explicit causal markers (highest priority)
    for i, sentence in enumerate(sentences):
        has_marker, marker, split_point = detect_causal_relationship(sentence)
        
        if has_marker:
            before = sentence[:split_point].strip()
            after = sentence[split_point + len(marker):].strip()
            
            emotion_in_before, _ = emotion_keyword_in_text(before, emotion)
            emotion_in_after, _ = emotion_keyword_in_text(after, emotion)
            
            if emotion_in_after and before and len(before.split()) >= 3:
                pairs.append({
                    'emotion': emotion,
                    'cause': before,
                    'evidence': sentence,
                    'confidence': confidence * 0.95,  # Very high for explicit
                    'method': 'explicit_causal_marker'
                })
            elif emotion_in_before and after and len(after.split()) >= 3:
                pairs.append({
                    'emotion': emotion,
                    'cause': after,
                    'evidence': sentence,
                    'confidence': confidence * 0.95,
                    'method': 'explicit_causal_marker'
                })
            elif after and len(after.split()) >= 3:
                # Default: cause after marker
                pairs.append({
                    'emotion': emotion,
                    'cause': after,
                    'evidence': sentence,
                    'confidence': confidence * 0.85,
                    'method': 'causal_marker'
                })
    
    # Strategy 2: Context-aware window (improved)
    for i, sentence in enumerate(sentences):
        has_emotion, trigger = emotion_keyword_in_text(sentence, emotion)
        
        if has_emotion and len(sentence.split()) >= 4:
            # Previous sentence (common for cause)
            if i > 0 and len(sentences[i-1].split()) >= 4:
                prev_sentence = sentences[i - 1]
                # Higher confidence if previous sentence doesn't have emotion
                prev_has_emotion, _ = emotion_keyword_in_text(prev_sentence, emotion)
                conf_multiplier = 0.75 if not prev_has_emotion else 0.55
                
                pairs.append({
                    'emotion': emotion,
                    'cause': prev_sentence,
                    'evidence': f"{prev_sentence} {sentence}",
                    'confidence': confidence * conf_multiplier,
                    'method': 'contextual_precedence'
                })
    
    # Strategy 3: Enhanced clause extraction
    for sentence in sentences:
        has_emotion, trigger = emotion_keyword_in_text(sentence, emotion)
        
        if has_emotion:
            clauses = extract_clauses(sentence)
            
            for clause in clauses:
                clause_text = clause['text'].strip()
                # Only consider substantial clauses
                if len(clause_text.split()) >= 3:
                    # Check if clause doesn't contain emotion word
                    clause_has_emotion = trigger and trigger in clause_text.lower()
                    
                    if not clause_has_emotion:
                        pairs.append({
                            'emotion': emotion,
                            'cause': clause_text,
                            'evidence': sentence,
                            'confidence': confidence * 0.70,
                            'method': 'clause_extraction'
                        })
    
    # Strategy 4: Sentence-level emotion intensity (new)
    # Find sentences with action verbs but without emotion
    nlp = get_spacy_model()
    for sentence in sentences:
        has_emotion, _ = emotion_keyword_in_text(sentence, emotion)
        
        if not has_emotion and len(sentence.split()) >= 4:
            doc = nlp(sentence)
            # Check for action verbs (indicators of events/causes)
            has_action = any(token.pos_ == 'VERB' and token.dep_ in ['ROOT', 'ccomp', 'xcomp'] 
                           for token in doc)
            
            if has_action:
                pairs.append({
                    'emotion': emotion,
                    'cause': sentence,
                    'evidence': sentence,
                    'confidence': confidence * 0.60,
                    'method': 'action_based'
                })
    
    # Strategy 5: Fallback with full text
    if not pairs:
        pairs.append({
            'emotion': emotion,
            'cause': text if len(text.split()) <= 50 else text[:200] + '...',
            'evidence': text,
            'confidence': confidence * 0.35,
            'method': 'fallback'
        })
    
    # Advanced deduplication and ranking
    pairs = deduplicate_pairs(pairs)
    pairs = sorted(pairs, key=lambda x: (x['confidence'], len(x['cause'].split())), reverse=True)
    
    # Return top 5 pairs (increased from 3)
    return pairs[:5]

def deduplicate_pairs(pairs):
    """
    Remove duplicate or highly similar pairs
    """
    if not pairs:
        return []
    
    unique_pairs = []
    seen_causes = set()
    
    for pair in pairs:
        cause_normalized = ' '.join(pair['cause'].lower().split())
        
        # Check if this cause is too similar to existing ones
        is_duplicate = False
        for seen_cause in seen_causes:
            # Simple similarity check
            if cause_normalized in seen_cause or seen_cause in cause_normalized:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_pairs.append(pair)
            seen_causes.add(cause_normalized)
    
    return unique_pairs

def format_ecpe_output(pairs):
    """
    Format ECPE pairs into clean JSON output
    """
    formatted = []
    
    for i, pair in enumerate(pairs):
        formatted.append({
            'pair_id': i + 1,
            'emotion': pair['emotion'],
            'cause': pair['cause'],
            'evidence': pair['evidence'],
            'confidence': round(pair['confidence'], 3),
            'extraction_method': pair.get('method', 'unknown')
        })
    
    return formatted

if __name__ == "__main__":
    # Test the ECPE module
    test_cases = [
        {
            'text': "I failed my exam because I didn't study enough. Now I feel so disappointed.",
            'emotion': 'sadness',
            'confidence': 0.95
        },
        {
            'text': "I am so happy today! I got accepted to my dream university.",
            'emotion': 'joy',
            'confidence': 0.92
        },
        {
            'text': "The traffic made me late for the meeting. I am so angry and frustrated.",
            'emotion': 'anger',
            'confidence': 0.88
        }
    ]
    
    print("Testing ECPE Module:\n")
    for test in test_cases:
        print(f"Text: {test['text']}")
        print(f"Emotion: {test['emotion']} (confidence: {test['confidence']})\n")
        
        pairs = extract_emotion_cause_pairs(test['text'], test['emotion'], test['confidence'])
        formatted = format_ecpe_output(pairs)
        
        print("Extracted Pairs:")
        for pair in formatted:
            print(f"  - Cause: {pair['cause']}")
            print(f"    Confidence: {pair['confidence']}")
            print(f"    Method: {pair['extraction_method']}\n")
        print("-" * 80 + "\n")

