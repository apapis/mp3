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

def prepare_prompt(transcripts):
    prompt = """Take a deep breath. Let's solve this step by step.

You need to analyze interrogation transcripts to find the specific street name where Professor Maj's institute is located. This is a critical task that requires careful consideration of all testimonies.

Here are the transcripts to analyze:
"""
    
    for transcript in transcripts:
        prompt += f"\n=== {transcript['name']}'s testimony ===\n{transcript['content']}\n"
        
    prompt += """
Now, let's think about this carefully:

1. First, let's identify all mentions of locations, buildings, or directions in the testimonies.
2. Let's analyze which testimonies seem more reliable and why.
3. Let's look for any patterns or corroborating details between different testimonies.
4. We need to find not just any location, but specifically the street where the institute is located.
5. Remember that some testimonies might be contradictory or misleading.

Based on this analysis, please provide the exact street name where Professor Maj's specific institute is located.

Think carefully and show your reasoning before providing the final answer."""

    return prompt

def load_transcripts():
    transcripts = []
    files = [f for f in os.listdir('.') if f.endswith('_transcript.txt')]
    
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            person_name = file.split('_')[0]
            content = f.read()
            transcripts.append({
                'name': person_name,
                'content': content
            })
    
    return transcripts

def main():
    # audio_files = find_audio_files()
    
    # if not audio_files:
    #     print("No audio files found!")
    #     return
    
    # print(f"Found {len(audio_files)} audio files:")
    # for file in audio_files:
    #     print(f"- {file}")
    
    # for file in audio_files:
    #     print(f"\nProcessing: {file}")
    #     transcription = transcribe_file(file)
        
    #     if transcription:
    #         output_file = f"{os.path.splitext(file)[0]}_transcript.txt"
    #         with open(output_file, 'w', encoding='utf-8') as txt_file:
    #             txt_file.write(transcription)
    #         print(f"Saved transcription to: {output_file}")

    print("\nPreparing analysis...")
    transcripts = load_transcripts()
    
    if not transcripts:
        print("No transcripts found!")
        return
        
    prompt = prepare_prompt(transcripts)
    print("\nPrepared prompt for analysis:")
    print(prompt) # Preview first 200 characters
    
    return prompt

if __name__ == "__main__":
    main()