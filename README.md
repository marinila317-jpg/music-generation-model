# Music Generation Model

This repository contains the source code and resources for developing a deep learning model for music generation. The project aims to explore modern architectures such as Transformers, VAEs, and Diffusion models for generating high-quality, expressive music.

## Project Structure

- `data/`: Contains datasets and data preprocessing scripts.
- `models/`: Contains model definitions and architectures (e.g., PyTorch modules).
- `training/`: Contains scripts for training the models.
- `generation/`: Contains scripts for generating music using trained models.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model experimentation.
- `docs/`: Documentation and research notes.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/music_generation_model.git
   cd music_generation_model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the data:
   Follow the instructions in `data/README.md` to download and preprocess the dataset.

4. Train the model:
   ```bash
   python training/train.py --config configs/default.yaml
   ```

5. Generate music with MusicGen (Command Line):

   ```bash
   python generation/generate.py --prompt "a cheerful and upbeat pop song" --duration 10 --model_version small
   ```
   You can specify different prompts, duration, and model versions (small, medium, large, melody).


## Web Interface (Gradio)

For a more interactive experience, you can use the Gradio web interface:

1.  **Run the Gradio app:**
    ```bash
    python app.py
    ```
2.  Open the provided local or public URL in your browser.

## Fine-tuning MusicGen

To fine-tune the MusicGen model on your custom dataset (e.g., for a specific genre like "Dark Ethno Folk"), follow these steps:

1.  **Prepare your dataset:**
    *   Create a directory (e.g., `data/my_custom_music/`).
    *   Place your audio files (preferably `.wav` format) into this directory.
    *   Create a `metadata.csv` file inside your dataset directory. This file should contain at least two columns: `path` (relative path to the audio file from the dataset directory) and `text` (a descriptive caption for the audio).
    *   Example `data/my_custom_music/metadata.csv`:
        ```csv
        path,text,genre
        track1.wav,"A dark ethno folk track with deep drums and mystical chanting",folk
        track2.wav,"Ambient soundscape with tribal percussion and deep bass",ambient
        ```

2.  **Run the fine-tuning script:**
    ```bash
    python training/finetune.py --dataset_dir data/my_custom_music/ --output_dir checkpoints/finetuned_musicgen/
    ```
    *   `--dataset_dir`: Path to your custom dataset directory containing audio files and `metadata.csv`.
    *   `--output_dir`: Directory where the fine-tuned model checkpoint will be saved.

**Important Notes on Fine-tuning:**
*   The actual fine-tuning process for AudioCraft/MusicGen is complex and requires a dedicated training setup, often involving distributed training and specific data preprocessing pipelines (e.g., tokenization with EnCodec). The provided `finetune.py` script is a conceptual outline.
*   You will need a GPU with sufficient VRAM for fine-tuning, especially for larger models.
*   Refer to the official AudioCraft documentation and examples for a full implementation of custom model training.

## Architectures Explored

This project initially includes a basic LSTM model and is being expanded to integrate more advanced architectures:

- **MusicGen (via AudioCraft)**: State-of-the-art text-to-music generation from Meta, allowing generation from text prompts.
- **Transformers**: Auto-regressive models suitable for symbolic music generation, capturing long-range dependencies.
- **Variational Autoencoders (VAEs)**: Useful for learning latent representations of musical segments, enabling interpolation and style transfer.
- **Diffusion Models**: Advanced models for high-fidelity audio generation, capable of producing realistic soundscapes and musical pieces.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
