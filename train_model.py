import pandas as pd
import re
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# === Config ===
DATA_DIR = r"C:\Users\ADMIN\Desktop\Buildathon\data"
REAL_EMAIL_FILE = "Phishing_Email.csv"
AI_EMAIL_FILE = "ai_generated_emails.csv"

# === Load datasets ===
df_real = pd.read_csv(os.path.join(DATA_DIR, REAL_EMAIL_FILE))
df_ai = pd.read_csv(os.path.join(DATA_DIR, AI_EMAIL_FILE))

# === Normalize Labels ===
df_real['Email Type'] = df_real['Email Type'].str.strip().str.lower().replace({
    'Phishing Email': 'human_written_phishing',
    'Safe Email': 'safe_email'
})
df_ai['Email Type'] = df_ai['Email Type'].str.strip().str.lower().replace({
    'ai_generated_phishing': 'ai_generated_phishing',
    'ai_generated_safe': 'ai_generated_safe',
})

# === Combine datasets ===
df = pd.concat([df_real, df_ai], ignore_index=True)

# Normalize Labels after concatenation
df['Email Type'] = df['Email Type'].str.strip().str.lower().replace({
    'phishing email': 'human_written_phishing',
    'safe email': 'safe_email',
    'ai_generated_phishing': 'ai_generated_phishing',
    'ai_generated_safe': 'ai_generated_safe'
})


# === Text cleaning function ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@[\w]+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === Preprocessing ===
df.dropna(subset=["Email Text", "Email Type"], inplace=True)
df.drop_duplicates(inplace=True)
df['clean_text'] = df['Email Text'].apply(clean_text)

# === Label distribution check ===
print("\n📊 Final dataset label distribution:")
print(df['Email Type'].value_counts())

# === Features and Labels ===
X = df['clean_text']
y = df['Email Type']

# === Train/Test Split (Stratified) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === Class Weights (Adjusted) ===
class_weight_dict = {
    'safe_email': 1,
    'human_written_phishing': 2,
    'ai_generated_safe': 1,
    'ai_generated_phishing': 2
}

# === Train Logistic Regression Model ===
model = LogisticRegression(max_iter=1000, class_weight=class_weight_dict)
model.fit(X_train_vec, y_train)

# === Check Model Classes ===
print(f"Model classes: {model.classes_}")

# === Evaluation ===
y_pred = model.predict(X_test_vec)

print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f}")

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
plt.figure(figsize=(10, 7))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("📊 Confusion Matrix")
plt.show()

# === Save model and vectorizer ===
os.makedirs("model", exist_ok=True)
with open("model/phishing_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save label list
with open("model/labels.pkl", "wb") as f:
    pickle.dump(model.classes_, f)

print("\n✅ Model, vectorizer, and label list saved in the 'model' folder.")

# === Prediction Function ===
def predict_email_type(email_text):
    email_cleaned = clean_text(email_text)
    email_vec = vectorizer.transform([email_cleaned])
    prediction = model.predict(email_vec)[0]
    print(f"\n🔍 Predicted label: {prediction}")  # <-- Helpful for debugging
    explanation = {
        'ai_generated_phishing': "📧 This is an AI-generated phishing email.",
        'ai_generated_safe': "📧 This is an AI-generated safe email.",
        'human_written_phishing': "📧 This is a human-written phishing email.",
        'safe_email': "📧 This is a human-written safe email.",
    }
    result = explanation.get(prediction, "❗ Unable to classify the email.")
    print(result)
    return result


# === Example Usage ===
email_text = """  
Hi Team,  

Just a quick reminder about tomorrow’s meeting at 10 AM in Conference Room B.  

Agenda:  
1. Project updates  
2. Q2 goals review  
3. Open discussion  

Bring your laptops!  

Thanks,  
John  
"""  

predict_email_type(email_text)  