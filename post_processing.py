import pedalboard
from pedalboard.io import AudioFile
import numpy as np

def apply_post_processing(audio_data: np.ndarray, sample_rate: int, effect_preset: str = "default") -> np.ndarray:
    """
    Applies a set of audio effects to the given audio data using Pedalboard.

    Args:
        audio_data (np.ndarray): The audio waveform as a NumPy array.
        sample_rate (int): The sample rate of the audio.
        effect_preset (str): The preset of effects to apply. Options: "default", "dark_ethno_folk", "none".

    Returns:
        np.ndarray: The processed audio waveform.
    """
    board = pedalboard.Pedalboard()

    if effect_preset == "dark_ethno_folk":
        # Custom preset for Dark Ethno Folk style
        board.append(pedalboard.Reverb(room_size=0.7, damping=0.5, wet_level=0.3, dry_level=0.7))
        board.append(pedalboard.Compressor(threshold_db=-18, ratio=4.0, attack_ms=10, release_ms=100))
        board.append(pedalboard.Gain(gain_db=3))
        # A subtle EQ to enhance low-mids and highs for a 'dark' and 'ethno' feel
        board.append(pedalboard.BandpassFilter(cutoff_frequency_hz=150, q=0.7))
        board.append(pedalboard.HighpassFilter(cutoff_frequency_hz=50))
        board.append(pedalboard.LowpassFilter(cutoff_frequency_hz=10000))
        print("Applying 'Dark Ethno Folk' post-processing effects.")
    elif effect_preset == "default":
        # A general purpose preset
        board.append(pedalboard.Reverb(room_size=0.5, damping=0.3, wet_level=0.2, dry_level=0.8))
        board.append(pedalboard.Compressor(threshold_db=-20, ratio=3.0, attack_ms=5, release_ms=80))
        board.append(pedalboard.Gain(gain_db=2))
        print("Applying 'default' post-processing effects.")
    elif effect_preset == "none":
        print("No post-processing effects applied.")
        return audio_data
    else:
        print(f"Unknown effect preset: {effect_preset}. Applying 'default' effects.")
        board.append(pedalboard.Reverb(room_size=0.5, damping=0.3, wet_level=0.2, dry_level=0.8))
        board.append(pedalboard.Compressor(threshold_db=-20, ratio=3.0, attack_ms=5, release_ms=80))
        board.append(pedalboard.Gain(gain_db=2))

    # Pedalboard expects float32 audio, so convert if necessary
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    processed_audio = board(audio_data, sample_rate)
    return processed_audio


if __name__ == "__main__":
    # Example usage (requires an input audio file)
    # You would typically call this from your generation script.
    # For demonstration, let's create a dummy audio file and process it.
    print("Running example post-processing...")
    dummy_sample_rate = 44100
    dummy_duration = 5 # seconds
    dummy_audio = np.random.uniform(-0.5, 0.5, size=(dummy_sample_rate * dummy_duration,)).astype(np.float32)

    input_dummy_path = "dummy_input.wav"
    output_dummy_path = "dummy_output_processed.wav"

    # Save dummy audio to file
    with AudioFile(input_dummy_path, 'w', dummy_sample_rate, dummy_audio.shape[0] if dummy_audio.ndim == 1 else dummy_audio.shape[1]) as f:
        f.write(dummy_audio)

    # Read dummy audio from file
    with AudioFile(input_dummy_path) as f:
        audio = f.read(f.frames)
        samplerate = f.samplerate

    # Apply effects
    processed_audio = apply_post_processing(audio, samplerate, effect_preset="dark_ethno_folk")

    # Write processed audio to file
    with AudioFile(output_dummy_path, 'w', samplerate, processed_audio.shape[0] if processed_audio.ndim == 1 else processed_audio.shape[1]) as f:
        f.write(processed_audio)

    print(f"Dummy audio processed and saved to {output_dummy_path}")
    os.remove(input_dummy_path) # Clean up dummy input
