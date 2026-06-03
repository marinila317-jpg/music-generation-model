import gradio as gr
import os
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Load the MusicGen model once when the app starts
# Using 'small' model for faster loading and less memory usage for a demo
# You can change this to 'medium', 'large', or 'melody' if you have enough VRAM
print("Loading MusicGen model (small) for Gradio app...")
model = MusicGen.get_pretrained("small")

def generate_music_gradio(prompt, duration, model_version):
    """
    Generates music using MusicGen from AudioCraft based on a text prompt
    and returns the path to the generated audio file.
    """
    global model # Use the globally loaded model

    # If the user selects a different model version, reload it
    if model.name != model_version:
        print(f"Reloading MusicGen model to {model_version}...")
        model = MusicGen.get_pretrained(model_version)

    model.set_generation_params(duration=duration)

    print(f"Generating music for prompt: \'{prompt}\' with duration {duration}s using {model_version} model...")
    wav = model.generate([prompt])

    output_dir = "generated_music"
    os.makedirs(output_dir, exist_ok=True)
    # Sanitize prompt for filename
    safe_prompt = "_".join(prompt.split()[:5]) # Take first 5 words for filename
    output_path = os.path.join(output_dir, f"musicgen_output_{safe_prompt}_{duration}s.wav")

    for idx, one_wav in enumerate(wav):
        audio_write(output_path, one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
        print(f"Generated music saved to {output_path}")
        break # Only save the first generated sample for simplicity

    return output_path

# Define the Gradio interface
iface = gr.Interface(
    fn=generate_music_gradio,
    inputs=[
        gr.Textbox(label="Text Prompt", placeholder="e.g., 'Dark Ethno Folk with mystical atmosphere and heavy drums'"),
        gr.Slider(minimum=5, maximum=30, value=10, step=1, label="Duration (seconds)"),
        gr.Dropdown(choices=["small", "medium", "large", "melody"], value="small", label="MusicGen Model Version")
    ],
    outputs=gr.Audio(label="Generated Music"),
    title="Music Generation with MusicGen (AudioCraft)",
    description="Enter a text prompt to generate music using Meta's MusicGen model. Please note that 'medium' and 'large' models require significant GPU memory."
)

if __name__ == "__main__":
    iface.launch(share=True) # share=True creates a public link for easy sharing
