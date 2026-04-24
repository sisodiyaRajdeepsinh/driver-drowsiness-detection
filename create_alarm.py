import wave
import math
import struct

def generate_alarm_tone(filename='alarm.wav', duration=1.0, freq=1000, volume=0.5, sample_rate=44100):
    n_samples = int(sample_rate * duration)
    
    with wave.open(filename, 'w') as wav_file:
        # Set parameters: 1 channel, 2 bytes per sample, sample rate
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        data = []
        for i in range(n_samples):
            # Modulate volume to create a "beeping" effect (4Hz beep)
            modulation = 1.0 if (int(i / (sample_rate / 4)) % 2 == 0) else 0.0
            
            # Generate sine wave
            value = int(volume * 32767.0 * modulation * math.sin(2.0 * math.pi * freq * i / sample_rate))
            data.append(struct.pack('<h', value))
            
        wav_file.writeframes(b''.join(data))
    print(f"Generated {filename}")

if __name__ == "__main__":
    generate_alarm_tone()
