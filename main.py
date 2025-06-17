from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import google.generativeai as genai
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=gemini_api_key)

def extract_video_id(url: str) -> str | None:
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(video_id: str) -> str:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([item['text'] for item in transcript])
    except (TranscriptsDisabled, NoTranscriptFound):
        return "Transcript is not available for this video."
    except Exception as e:
        return f"Error fetching transcript: {e}"

def summarize_text(text: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
    Summarize the following YouTube video transcript into a concise summary with:
    - Key Takeaways
    - Main Points Discussed
    - Conclusion

    Keep the language simple and clean.

    Transcript:
    {text}
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating summary: {e}"

def clean_gemini_output(raw_text: str) -> str:
    cleaned = re.sub(r"\*\*(.*?)\*\*", lambda m: m.group(1).upper(), raw_text)
    cleaned = re.sub(r"^\s*\*\s*", "- ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\*", "", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

def generate_summary_from_url(youtube_url: str) -> str:
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return "❌ Invalid YouTube URL."

    transcript = get_transcript(video_id)
    if "Transcript is not available" in transcript or "Error" in transcript:
        return transcript

    summary = summarize_text(transcript)
    if summary.startswith("Error"):
        return summary

    return clean_gemini_output(summary)

if __name__ == "__main__":
    url = input("Enter the YouTube URL: ").strip()
    print("\n📄 Video Summary:\n")
    print(generate_summary_from_url(url))
