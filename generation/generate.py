import os
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def generate_music_with_musicgen(prompt, duration=8, output_dir="outputs", model_version="small"):
    """
    Generates music using MusicGen from AudioCraft based on a text prompt.

    Args:
        prompt (str): The text description of the music to generate.
        duration (int): The duration of the generated music in seconds.
        output_dir (str): Directory to save the generated audio.
        model_version (str): The version of the MusicGen model to use (e.g., "small", "medium", "large").
    """
    print(f"Loading MusicGen model ({model_version})...")
    model = MusicGen.get_pretrained(model_version)
    model.set_generation_params(duration=duration)

    print(f"Generating music for prompt: '{prompt}'...")
    wav = model.generate([prompt])

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"musicgen_output_{prompt.replace(' ', '_')}.wav")

    for idx, one_wav in enumerate(wav):
        # You can save multiple samples if generated
        audio_write(output_path, one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
        print(f"Generated music saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate music using MusicGen from AudioCraft.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for music generation.")
    parser.add_argument("--duration", type=int, default=8,
                        help="Duration of the generated music in seconds.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save the generated music.")
    parser.add_argument("--model_version", type=str, default="small",
                        choices=["small", "medium", "large", "melody"],
                        help="Version of the MusicGen model to use.")
    args = parser.parse_args()

    generate_music_with_musicgen(args.prompt, args.duration, args.output_dir, args.model_version)
