from flask import Flask, render_template, request
import pickle
import re
import os

app = Flask(__name__)

# Load model and vectorizer with error handling
try:
    model = pickle.load(open("model/phishing_model.pkl", "rb"))
    vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {str(e)}")

def clean_text(text):
    text = str(text).lower()
    
    # Keep emojis (AI uses them more frequently)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Still remove URLs
    text = re.sub(r'\@\w+', '', text)  # Remove mentions
    
    # Preserve special chars/emojis for AI detection
    text = re.sub(r'(?<!\w)[^\w\s\.,!?@#\U0001F300-\U0001F6FF](?!\w)', '', text)
    
    return text.strip()

def predict_email_type(email_text):
    email_cleaned = clean_text(email_text)
    email_vec = vectorizer.transform([email_cleaned])
    prediction = model.predict(email_vec)[0]

    # Enhanced explanation dictionary
    explanation = {
        'ai_generated_phishing': "⚠️ AI Phishing: Sophisticated scam with generated text",
        'ai_generated_spam': "⚠️ AI Spam: Bulk generated promotional content",
        'human_written_phishing': "⛔ Human Phishing: Crafted by scammers",
        'safe_email': "✅ Legitimate email",
        # Add other classes as needed
    }

    # Additional AI detection heuristics
    if prediction in ['human_written_phishing', 'ai_generated_phishing']:
        if sum(1 for c in email_text if ord(c) > 128) > 3:  # Unicode chars (emojis)
            return "⚠️ Likely AI-Generated (excessive emojis/special chars)"
    
    return explanation.get(prediction, "Unknown email type")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    email_text = ""
    
    if request.method == "POST":
        email_text = request.form.get("email", "")
        if email_text.strip():
            prediction = predict_email_type(email_text)
    
    return render_template("index.html", 
                         prediction=prediction,
                         email_text=email_text)

if __name__ == "__main__":
    # Use environment variable for port or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)  # debug=False for production