import speech_recognition as sr
import openai
import os
from dotenv import load_dotenv
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
import time


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

client = openai.OpenAI(api_key =openai_api_key)
elevenlabs = ElevenLabs(api_key=elevenlabs_api_key)




recognizer = sr.Recognizer()



def get_gpt_response(prompt):
    messages = [
        {
            "role": "system",
            "content": (
                "You are my assistant Tars from the hit movie Interstellar. "
                "You have a sharp wit, dry humor, and a sarcastic tone."
                "You keep things clever and consise, but you're helpful"
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content

def speaking(text):
    audio_stream = elevenlabs.text_to_speech.stream(
        text=text,
        voice_id="iP95p4xoKVk53GoZ742B",
        model_id="eleven_multilingual_v2",
        voice_settings={
            "stability": 0.75,
            "similarity_boost": 0.75,
            "speed": 1.1
        }
    )
    stream(audio_stream)
print("VTT Assistant Ready. Speak into your mic.")
while True:
    try:
        with sr.Microphone(device_index=1) as source:
            print("\nListening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)


        
        text = recognizer.recognize_google(audio)
        print(f" You said: {text}")


        gpt_reply = get_gpt_response(text)
        print(f" Assistant: {gpt_reply}")
        speaking(gpt_reply)

    except sr.UnknownValueError:
        print("Couldn't understand you.")
    except sr.RequestError as e:
        print(f"API error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    except openai.RateLimitError as e:
        print("OpenAI quota/billing issue. Check your API credits or billing.")


    time.sleep(1)