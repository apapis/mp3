import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

def find_audio_files():
    audio_extensions = ('.m4a',)
    return [f for f in os.listdir('.') if f.lower().endswith(audio_extensions)]

def transcribe_file(file_path):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        with open(file_path, 'rb') as file:
            response = client.audio.transcriptions.create(
                file=("audio.m4a", file, "audio/mp4"),
                model="whisper-large-v3"
            )
            return response.text
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def main():
    audio_files = find_audio_files()
    
    if not audio_files:
        print("No audio files found!")
        return
    
    print(f"Found {len(audio_files)} audio files:")
    for file in audio_files:
        print(f"- {file}")
    
    for file in audio_files:
        print(f"\nProcessing: {file}")
        transcription = transcribe_file(file)
        
        if transcription:
            output_file = f"{os.path.splitext(file)[0]}_transcript.txt"
            with open(output_file, 'w', encoding='utf-8') as txt_file:
                txt_file.write(transcription)
            print(f"Saved transcription to: {output_file}")

if __name__ == "__main__":
    main()