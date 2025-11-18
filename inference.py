import whisper
import time
import torch
import pandas as pd


model_names = ["tiny", "base", "small", "medium", "large", "turbo"]
# model_names = ["turbo"]
audio_path = "/home/fan/project/whisper/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac"

num_runs = 1
results = []

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Detected device: {device}")

audio = whisper.load_audio(audio_path)
audio = whisper.pad_or_trim(audio)

for name in model_names:
    print(f"\n--- Testing model: {name} ---")
    model = whisper.load_model(name, device=device)
    print(model.dims)
    
    # mel spectrogram
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)


    # warm-up 
    _ = whisper.decode(model, mel, whisper.DecodingOptions())

    # inference
    times = []
    for i in range(num_runs):
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.time()
        result = whisper.decode(model, mel, whisper.DecodingOptions())
        torch.cuda.synchronize() if device == "cuda" else None
        t1 = time.time()
        times.append(t1 - t0)
        print(result.text)
        # print(f"Run {i+1}: {times[-1]:.2f}s")

    avg_time = sum(times) / len(times)
    print(f"Average decoding time ({name}): {avg_time:.2f}s")

    results.append({
        "Model": name,
        "Parameters": model.dims.n_text_layer * model.dims.n_audio_layer,  
        # "Average Latency (s)": round(avg_time, 2),
        "Device": device,
    })


df = pd.DataFrame(results)
print("\n===== Latency Benchmark Results =====")
print(df.to_markdown(index=False))
