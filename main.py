import os
import re
from datetime import timedelta
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv
from openai import OpenAI
import yt_dlp
from davia import Davia

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

app = Davia()


def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    return str(td)[2:] if td < timedelta(hours=1) else str(td)


def extract_video_id(url: str) -> str | None:
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def get_transcript(video_id: str) -> str:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return "\n".join([f"{format_timestamp(item['start'])} - {item['text']}" for item in transcript])
    except (TranscriptsDisabled, NoTranscriptFound):
        return "Transcript is not available for this video."
    except Exception as e:
        return f"Error fetching transcript: {e}"


def get_video_metadata(url: str) -> tuple[str, str]:
    try:
        ydl_opts = {'quiet': True, 'skip_download': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'No title found')
            duration = str(timedelta(seconds=info.get('duration', 0)))
            return title, duration
    except Exception:
        return "Unknown Title", "Unknown Duration"


def summarize_text_with_openai(text: str) -> str:
    prompt = f"""
You are a helpful assistant. Return a:
1. Beginner-friendly summary.
2. List of key highlights in bullet points.
(Plain text, no markdown headers.)

Transcript:
{text}
"""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You summarize YouTube transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary: {e}"


def clean_response_output(raw_text: str) -> tuple[str, list[str]]:
    raw_text = raw_text.replace("**", "").strip()
    raw_text = re.sub(r"\bKey\s*\n\s*Highlights\b", "Key Highlights", raw_text, flags=re.IGNORECASE)
    parts = re.split(r"(?i)key highlights\s*[:\-]?", raw_text)
    summary_part = parts[0].strip()
    highlight_text = parts[1].strip() if len(parts) > 1 else ""
    highlights = re.findall(r"[-•]\s+(.*)", highlight_text)
    return summary_part, highlights


@app.task
def action_summarize_youtube(
    youtube_url: str
) -> dict:
    """
    Summarize a YouTube video.

    Args:
      youtube_url: Full URL of the YouTube video.

    Returns:
      Dict including:
        - video_url: embed URL for iframe
        - video_title: Title
        - video_duration: Duration (HH:MM:SS)
        - transcript: Full transcript with timestamps
        - summary: Clean summary paragraph
        - highlights: List of key bullet points
    """
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return {"error": "❌ Invalid YouTube URL."}

    transcript = get_transcript(video_id)
    if "Transcript is not available" in transcript or transcript.startswith("Error"):
        return {"error": transcript}

    summary_text = summarize_text_with_openai(transcript)
    if summary_text.startswith("Error"):
        return {"error": summary_text}

    summary, highlights = clean_response_output(summary_text)
    title, duration = get_video_metadata(youtube_url)

    return {
        "video_url": f"https://www.youtube.com/embed/{video_id}",
        "video_title": title,
        "video_duration": duration,
        "transcript": transcript,
        "summary": summary,
        "highlights": highlights
    }


if __name__ == "__main__":
    app.run()
