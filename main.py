import os
from groq import Groq
from dotenv import load_dotenv
from openai import OpenAI
import requests
import json

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

1. Extract every detail about the professor's daily routine and movement patterns
2. Pay special attention to descriptions of the institute's surroundings
3. Note any mentions of public transport routes or landmarks near the institute
4. Focus on consistent details that appear in multiple testimonies
5. Look for specific academic departments or fields of study mentioned
6. Use your knowledge to match these details with real locations



Based on this analysis, please provide the exact street name where Professor Maj's specific institute is located.

Think carefully and show your reasoning before providing the final answer.

Provide your final answer in the following JSON format:
{
    "thinking": "Your detailed step-by-step analysis and reasoning process",
    "answer": "The exact street name where the institute is located (ONLY STREET NAME)"
}
"""

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

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def send_to_openai(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role":"user", "content":prompt}],
            temperature=0.5
        )

        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error during OpenAi API call: {str(e)}")
        return None

def send_answer(street):
    report_api_key = os.getenv('REPORT_API_KEY')
    report_url = os.getenv('REPORT_URL')
    
    pyload = {
        "task":"mp3",
        "apikey": report_api_key,
        "answer": street
    }

    try:
        response = requests.post(report_url, json=pyload)
        result = response.json()

        if result.get('code')==0:
            print("Success:", result.get('message'))
            return True
        else:
            print("Error:", result.get('message'))
            return False
        
    except requests.RequestException as e:
        print(f"Error sending answer: {e}")
        return False

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

    print("\nPreparing analysis...")
    transcripts = load_transcripts()
    
    if not transcripts:
        print("No transcripts found!")
        return
        
    prompt = prepare_prompt(transcripts)
    print("\nPrepared prompt for analysis:")
    print(prompt)

    print("\nSending to OpenAI...")
    
    analysis = send_to_openai(prompt)

    if analysis:
        try:
            # Parse the JSON response
            result = json.loads(analysis)
            
            print("\nAnalysis reasoning:")
            print(result['thinking'])
            
            print("\nFound street:")
            print(result['answer'])
            
            # Send the answer
            print("\nSending answer to API...")
            if send_answer(result['answer']):
                print("Answer submitted successfully!")
            else:
                print("Failed to submit answer.")
            
            # Save full analysis
            with open('analysis_result.txt', 'w', encoding='utf-8') as f:
                f.write(analysis)
                
        except json.JSONDecodeError as e:
            print(f"Error parsing OpenAI response as JSON: {e}")
            print("Raw response:", analysis)

if __name__ == "__main__":
    main()