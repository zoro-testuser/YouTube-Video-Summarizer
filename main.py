from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import google.generativeai as genai
from dotenv import load_dotenv
import os
import re
from davia import Davia

app = Davia()

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=gemini_api_key)


@app.task
def extract_video_id(url: str) -> str | None:
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None


@app.task
def get_transcript(video_id: str) -> str:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([item['text'] for item in transcript])
    except (TranscriptsDisabled, NoTranscriptFound):
        return "Transcript is not available for this video."
    except Exception as e:
        return f"Error fetching transcript: {e}"


@app.task
def summarize_text(text: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
    Summarize the following YouTube video transcript into a well-organized response with:
    - A clean, beginner-friendly Summary
    - A bullet list of Highlights (key takeaways)
    
    Format clearly and keep the language simple.
 
    Transcript:
    {text}
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating summary: {e}"

@app.task
def clean_gemini_output(raw_text: str) -> str:
    cleaned = re.sub(r"\*\*(.*?)\*\*", lambda m: m.group(1).upper(), raw_text)
    cleaned = re.sub(r"^\s*\*\s*", "- ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\*", "", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


@app.task
def generate_summary_from_url(youtube_url: str) -> dict:
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return {"error": "❌ Invalid YouTube URL."}

    transcript = get_transcript(video_id)
    if "Transcript is not available" in transcript or "Error" in transcript:
        return {"error": transcript}

    summary_text = summarize_text(transcript)
    if summary_text.startswith("Error"):
        return {"error": summary_text}

    cleaned = clean_gemini_output(summary_text)

    highlights = re.findall(r"^- (.+)", cleaned, re.MULTILINE)
    summary_only = re.sub(r"^- .+\n?", "", cleaned, flags=re.MULTILINE).strip()

    return {
        "video_url": f"https://www.youtube.com/embed/{video_id}",
        "transcript": transcript,
        "summary": summary_only,
        "highlights": highlights
    }


if __name__ == "__main__":
    app.run()
