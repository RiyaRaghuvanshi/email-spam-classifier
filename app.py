from pathlib import Path
import re
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "spam_classifier.pkl"

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email Spam Classifier")
st.write(
    "Paste an email message below to check whether it looks like spam "
    "and assess its security risk."
)

# Check whether the trained model exists
if not MODEL_PATH.exists():
    st.warning(
        "The model has not been trained yet. "
        "Run `python train.py` in your terminal, then refresh this page."
    )
    st.stop()

model = joblib.load(MODEL_PATH)


# Spam Risk Assessment and Security Suggestions
def assess_risk(email):
    text = email.lower()
    score = 0
    reasons = []
    suggestions = []

    # Check for urgent language
    urgency_words = [
        "urgent", "immediately", "act now",
        "limited time", "expires today",
        "final warning", "account will be blocked"
    ]

    if any(word in text for word in urgency_words):
        score += 1
        reasons.append("Contains urgent or threatening language.")
        suggestions.append(
            "Do not rush. Verify the message through the "
            "company's official website or phone number."
        )

    # Check for sensitive information requests
    sensitive_phrases = [
        "verify your password", "enter your password",
        "share your otp", "provide your otp",
        "bank details", "credit card details",
        "verify your account", "confirm your identity"
    ]

    if any(phrase in text for phrase in sensitive_phrases):
        score += 2
        reasons.append(
            "May be asking for sensitive personal information."
        )
        suggestions.append(
            "Never share your password, OTP, or banking details "
            "through an email."
        )

    # Check for prize or money-related claims
    prize_phrases = [
        "you won", "claim your prize",
        "free gift", "lottery winner",
        "congratulations", "cash reward"
    ]

    if any(phrase in text for phrase in prize_phrases):
        score += 1
        reasons.append(
            "Contains an unusual prize or money-related claim."
        )
        suggestions.append(
            "Be careful with unexpected prizes or rewards. "
            "Do not pay fees to claim an unknown prize."
        )

    # Find links in the email
    urls = re.findall(r'https?://[^\s<>"\']+', text)

    suspicious_domains = [
        "bit.ly", "tinyurl.com", "t.co",
        "goo.gl", "is.gd"
    ]

    if any(
        domain in url
        for url in urls
        for domain in suspicious_domains
    ):
        score += 2
        reasons.append("Contains a shortened link.")
        suggestions.append(
            "Avoid clicking shortened links. Visit the official "
            "website by typing its address yourself."
        )

    # Check for unusual URL patterns
    unusual_url = any(
        "xn--" in url
        or re.search(
            r'https?://\d{1,3}(?:\.\d{1,3}){3}',
            url
        )
        for url in urls
    )

    if unusual_url:
        score += 2
        reasons.append("Contains an unusual link address.")
        suggestions.append(
            "Do not open unusual links. Check the website address "
            "carefully before visiting it."
        )

    # Decide the risk level
    if score >= 4:
        risk = "HIGH"
    elif score >= 2:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    if not reasons:
        reasons.append(
            "No suspicious patterns from the current checks were found."
        )
        suggestions.append(
            "Continue to be careful with unexpected attachments, "
            "links, and requests for personal information."
        )

    # Remove duplicate suggestions
    suggestions = list(dict.fromkeys(suggestions))

    return risk, reasons, suggestions


# Email input
email_text = st.text_area(
    "Email content",
    height=190,
    placeholder="Paste the email subject and body here..."
)


# Classify button
if st.button(
    "Classify Email",
    type="primary",
    use_container_width=True
):

    if not email_text.strip():
        st.error("Please enter an email message first.")

    else:
        # Machine Learning prediction
        prediction = model.predict([email_text])[0]

        st.subheader("📩 Email Classification")

        if prediction == "spam":
            st.error("🚨 This email is classified as SPAM.")
        else:
            st.success(
                "✅ This email is classified as NOT SPAM (HAM)."
            )

 
# Risk Assessment and Security Suggestions side by side
risk, reasons, suggestions = assess_risk(email_text)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🛡️ Spam Risk Assessment")

    if risk == "HIGH":
        st.error("🔴 HIGH RISK")
    elif risk == "MEDIUM":
        st.warning("🟠 MEDIUM RISK")
    else:
        st.success("🟢 LOW RISK")

    st.write("**Why is this email risky?**")

    for reason in reasons:
        st.write("•", reason)

with col2:
    st.subheader("🔐 Security Suggestions")

    st.write("**Stay safe with these tips:**")

    for suggestion in suggestions:
        st.info("💡 " + suggestion)



st.caption(
    "Educational demo only. Risk assessment is based on simple rules "
    "and does not guarantee that an email is safe. Do not rely on this "
    "model as the sole defense against phishing or malicious email."
)

