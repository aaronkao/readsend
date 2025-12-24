import os
import resend
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone
from src.daily_read import get_kindle_highlights, get_bible_verses, validate_env as validate_pinecone_env

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "aaron")
EMAIL_FROM = os.getenv("EMAIL_FROM", "onboarding@resend.dev") # Default Resend testing domain
EMAIL_TO = os.getenv("EMAIL_TO")

def validate_email_env():
    if not all([RESEND_API_KEY, EMAIL_TO]):
        print("Error: Missing email environment variables.")
        print("Please set RESEND_API_KEY and EMAIL_TO in .env")
        return False
    return True

def create_email_body(highlights):
    html = """
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <h2 style="color: #2c3e50;">Your Daily Highlights</h2>
        <p style="color: #7f8c8d;">Here are random highlights from your library for today:</p>
        <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
    """
    
    for i, h in enumerate(highlights, 1):
        meta = h['metadata']
        title = meta.get('title', 'Unknown Title')
        author = meta.get('author', 'Unknown Author')
        content = meta.get('content', '').strip()
        
        html += f"""
        <div style="margin-bottom: 30px; background-color: #f9f9f9; padding: 15px; border-radius: 5px;">
            <p style="font-size: 16px; font-style: italic; margin-top: 0;">"{content}"</p>
            <p style="font-size: 14px; color: #555; margin-bottom: 0; text-align: right;">
                — <strong>{title}</strong> <span style="color: #888;">({author})</span>
            </p>
        </div>
        """
        
    html += """
        <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
        <p style="font-size: 12px; color: #999; text-align: center;">
            Sent by ReadSend • <a href="#" style="color: #999;">Unsubscribe</a>
        </p>
    </body>
    </html>
    """
    return html

def send_email(subject, body):
    resend.api_key = RESEND_API_KEY
    
    params = {
        "from": EMAIL_FROM,
        "to": [EMAIL_TO],
        "subject": subject,
        "html": body,
    }
    
    try:
        email = resend.Emails.send(params)
        print(f"Email sent successfully! ID: {email.get('id')}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def main():
    # Validate environments
    # validate_pinecone_env exits on failure, so we just call it
    validate_pinecone_env()
    
    if not validate_email_env():
        return

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_INDEX_HOST)
    
    # Get highlights
    print("Fetching highlights...")
    kindle_highlights = get_kindle_highlights(index, count=5)
    bible_verses = get_bible_verses(index, count=2)
    highlights = kindle_highlights + bible_verses
    
    if not highlights:
        print("No highlights found.")
        return
    
    print(f"Fetched {len(kindle_highlights)} Kindle highlights and {len(bible_verses)} Bible verses.")
        
    # Create and send email
    today_str = datetime.now().strftime("%B %d, %Y")
    subject = f"Your Daily Highlights for {today_str}"
    body = create_email_body(highlights)
    
    print(f"Sending email to {EMAIL_TO}...")
    send_email(subject, body)

if __name__ == "__main__":
    main()
