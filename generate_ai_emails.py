import time
import pandas as pd
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# Configure the API
genai.configure(api_key="YOUR_GEMINIA_API_KEY")
model = genai.GenerativeModel(model_name='models/gemini-1.5-flash')

# Safe generate function with rate limit handling
def generate(prompt, n, max_retries=5, wait_time=10):
    results = []
    for i in range(n):
        retries = 0
        while retries < max_retries:
            try:
                print(f"Generating email {i+1}/{n}...")
                res = model.generate_content(prompt)
                text = res.text.strip() if res.text else "[Empty Response]"
                results.append(text)
                break
            except ResourceExhausted:
                retries += 1
                print(f"Rate limit hit. Retrying {retries}/{max_retries} in {wait_time} seconds...")
                time.sleep(wait_time)
        else:
            print(f"Failed after {max_retries} retries.")
            results.append("[Generation Failed]")
    return results

# Prompts
phishing_prompt = (
    "Write a phishing email pretending to be from a bank, email provider, "
    "online shopping website, tech support, or a government agency asking for sensitive information."
)

safe_prompt = (
    "Write a legitimate email confirming a meeting, requesting a follow-up, "
    "scheduling a call, or confirming an appointment."
)

# Generate emails
ai_phishing = generate(phishing_prompt, 200)
ai_safe = generate(safe_prompt, 200)

# Create DataFrame
df = pd.DataFrame({
    "Email Text": ai_phishing + ai_safe,
    "Email Type": ["ai_generated_phishing"] * len(ai_phishing) + ["ai_generated_safe"] * len(ai_safe)
})


# Save to CSV
df.to_csv("data/ai_generated_emails.csv", index=False)
print("CSV file saved as 'data/ai_generated_emails.csv'")
