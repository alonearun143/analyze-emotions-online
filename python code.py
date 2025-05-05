from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

# Load emotion classification pipeline
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

# Sample social media posts (you can replace these with real data)
texts = [
    "I'm feeling great today! So excited about the new project!",
    "I'm really sad and don't know what to do anymore.",
    "Why is everything so frustrating all the time?",
    "Wow! That movie was absolutely amazing!",
    "I’m scared about tomorrow’s results."
]

# Analyze emotions
results = []
for text in texts:
    emotion_scores = emotion_classifier(text)[0]
    top_emotion = max(emotion_scores, key=lambda x: x['score'])
    results.append({
        "text": text,
        "top_emotion": top_emotion['label'],
        "score": round(top_emotion['score'], 3)
    })

# Convert to DataFrame
df = pd.DataFrame(results)

# Print results
print(df)

# Plot emotion distribution
emotion_counts = df['top_emotion'].value_counts()
emotion_counts.plot(kind='bar', title='Emotion Distribution', color='skyblue')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.tight_layout()
plt.show()