# Orpheus TTS

#### Updates 🔥
- [5/2025] We've partnered with [Baseten](https://www.baseten.co/blog/canopy-labs-selects-baseten-as-preferred-inference-provider-for-orpheus-tts-model) to bring highly optimized inference to Orpheus at fp8 (more performant) and fp16 (full fidelity) inference. See code and docs [here](/additional_inference_options/baseten_inference_example/README.md).

- [4/2025] We release a [family of multilingual models](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba) in a research preview. We release a [training guide](https://canopylabs.ai/releases/orpheus_can_speak_any_language#training) that explains how we created these models in the hopes that even better versions in both the languages released and new languages are created. We welcome feedback and criticism as well as invite questions in this [discussion](https://github.com/canopyai/Orpheus-TTS/discussions/123) for feedback and questions.

## Overview
Orpheus TTS is a SOTA open-source text-to-speech system built on the Llama-3b backbone. Orpheus demonstrates the emergent capabilities of using LLMs for speech synthesis.

[Check out our original blog post](https://canopylabs.ai/model-releases)


https://github.com/user-attachments/assets/ce17dd3a-f866-4e67-86e4-0025e6e87b8a

## Abilities

- **Human-Like Speech**: Natural intonation, emotion, and rhythm that is superior to SOTA closed source models
- **Zero-Shot Voice Cloning**: Clone voices without prior fine-tuning
- **Guided Emotion and Intonation**: Control speech and emotion characteristics with simple tags
- **Low Latency**: ~200ms streaming latency for realtime applications, reducible to ~100ms with input streaming

## Models

We provide 2 English models, and additionally we offer the data processing scripts and sample datasets to make it very straightforward to create your own finetune.

1. [**Finetuned Prod**](https://huggingface.co/canopylabs/orpheus-tts-0.1-finetune-prod) – A finetuned model for everyday TTS applications

2. [**Pretrained**](https://huggingface.co/canopylabs/orpheus-tts-0.1-pretrained) – Our base model trained on 100k+ hours of English speech data

We also offer a family of multilingual models in a research release.

1. [**Multlingual Family**](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba) - 7 pairs of pretrained and finetuned models.

Orpheus TTS supports **multilingual fine-tuning and inference**, including Urdu (Nastaliq/Arabic script). See the Finetune and Prompting sections below for details on adding a new language like Urdu.

### Inference

#### Simple setup on Colab

We offer a standardised prompt format across languages, and these notebooks illustrate how to use our models in English.

1. [Colab For Tuned Model](https://colab.research.google.com/drive/1KhXT56UePPUHhqitJNUxq63k-pQomz3N?usp=sharing) (not streaming, see below for realtime streaming) – A finetuned model for everyday TTS applications.
2. [Colab For Pretrained Model](https://colab.research.google.com/drive/10v9MIEbZOr_3V8ZcPAIh8MN7q2LjcstS?usp=sharing) – This notebook is set up for conditioned generation but can be extended to a range of tasks.

#### One-click deployment on Baseten

Baseten is our [preferred inference partner](https://www.baseten.co/blog/canopy-labs-selects-baseten-as-preferred-inference-provider-for-orpheus-tts-model) for Orpheus. Get a dedicated deployment with real-time streaming on production-grade infrastructure [in one click on Baseten](https://www.baseten.co/library/orpheus-tts/).

#### Streaming Inference Example

1. Clone this repo
   ```bash
   git clone https://github.com/canopyai/Orpheus-TTS.git
   ```
2. Navigate and install packages
   ```bash
   cd Orpheus-TTS && pip install orpheus-speech # uses vllm under the hood for fast inference
   ```
   vllm pushed a slightly buggy version on March 18th so some bugs are being resolved by reverting to `pip install vllm==0.7.3` after `pip install orpheus-speech`
4. Run the example below (English):
   ```python
   from orpheus_tts import OrpheusModel
   import wave
   import time

   model = OrpheusModel(model_name ="canopylabs/orpheus-tts-0.1-finetune-prod", max_model_len=2048)
   prompt = '''Man, the way social media has, um, completely changed how we interact is just wild, right? Like, we're all connected 24/7 but somehow people feel more alone than ever. And don't even get me started on how it's messing with kids' self-esteem and mental health and whatnot.'''

   start_time = time.monotonic()
   syn_tokens = model.generate_speech(
      prompt=prompt,
      voice="tara",
      )

   with wave.open("output.wav", "wb") as wf:
      wf.setnchannels(1)
      wf.setsampwidth(2)
      wf.setframerate(24000)

      total_frames = 0
      chunk_counter = 0
      for audio_chunk in syn_tokens: # output streaming
         chunk_counter += 1
         frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
         total_frames += frame_count
         wf.writeframes(audio_chunk)
      duration = total_frames / wf.getframerate()

      end_time = time.monotonic()
      print(f"It took {end_time - start_time} seconds to generate {duration:.2f} seconds of audio")
   ```

   Or for **Urdu** (using a fine-tuned Urdu model and the `zia` voice):
   ```python
   from orpheus_tts import OrpheusModel
   import wave

   model = OrpheusModel(model_name="<YOUR_URDU_FINETUNED_MODEL>", max_model_len=2048)
   prompt = "zia: یہ ایک اردو متن ہے جسے بولا جائے گا۔"

   syn_tokens = model.generate_speech(
      prompt=prompt,
      voice="zia",
      )

   with wave.open("output_urdu.wav", "wb") as wf:
      wf.setnchannels(1)
      wf.setsampwidth(2)
      wf.setframerate(24000)
      for audio_chunk in syn_tokens:
         wf.writeframes(audio_chunk)
   ```

#### Setup Issues 

If you've cloned this repository and encounter a KV cache error or `max_model_len` property does not exist, use the local package instead of the installed PyPI version:

```python
import sys
sys.path.insert(0, 'orpheus_tts_pypi')
from orpheus_tts import OrpheusModel
```

This ensures you're using the repository code, which may have fixes not yet published to PyPI. See [#290](https://github.com/canopyai/Orpheus-TTS/issues/290) for more details.

#### Additional Functionality

1. Watermark your audio: Use Silent Cipher to watermark your audio generation; see [Watermark Audio Implementation](additional_inference_options/watermark_audio) for implementation.

2. For No GPU inference using Llama cpp see implementation [documentation](additional_inference_options/no_gpu/README.md) for implementation example


#### Prompting

1. The `finetune-prod` models: for the primary model, your text prompt is formatted as `{name}: I went to the ...`. The options for name in order of conversational realism (subjective benchmarks) are "tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe" for English - each language has different voices [see voices here] (https://canopylabs.ai/releases/orpheus_can_speak_any_language#info)). Our python package does this formatting for you, and the notebook also prepends the appropriate string. You can additionally add the following emotive tags: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`. For multilingual, see this [post](https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba) for supported tags.

   **Urdu (اردو) example** — use the `zia` voice prefix with Nastaliq/Arabic script text:
   ```
   zia: یہ ایک اردو متن ہے جسے بولا جائے گا۔
   ```
   When calling `generate_speech`, pass `voice="zia"` and the Urdu text as `prompt`. The engine will format the full prompt as `zia: <your Urdu text>` automatically.

2. The pretrained model: you can either generate speech just conditioned on text, or generate speech conditioned on one or more existing text-speech pairs in the prompt. Since this model hasn't been explicitly trained on the zero-shot voice cloning objective, the more text-speech pairs you pass in the prompt, the more reliably it will generate in the correct voice.


Additionally, use regular LLM generation args like `temperature`, `top_p`, etc. as you expect for a regular LLM. `repetition_penalty>=1.1`is required for stable generations. Increasing `repetition_penalty` and `temperature` makes the model speak faster.


## Automated Pipeline (single entry point)

`pipeline.py` at the repository root provides a **single command** for the entire workflow — raw data processing, speaker injection, SNAC audio encoding, tokenisation, and training — for both fine-tuning and pre-training, in English or any other language including Urdu.

### Install dependencies

```bash
pip install transformers datasets wandb peft flash_attn torch torchaudio snac
huggingface-cli login
wandb login
```

### Fine-tune (English or Urdu)

```bash
# English — speaker name read from the dataset's 'speaker' column automatically
python pipeline.py --phase finetune --dataset canopylabs/zac-sample-dataset

# Urdu — no 'speaker' column needed; pipeline adds "zia" prefix to every sample
python pipeline.py --phase finetune --dataset <your-urdu-dataset> --speaker zia

# LoRA (parameter-efficient) fine-tuning for Urdu
python pipeline.py --phase finetune --dataset <your-urdu-dataset> --speaker zia --lora
```

### Pre-train (speech-only or with interleaved text QA)

```bash
# Speech-only pre-training (recommended starting point for a new language)
python pipeline.py --phase pretrain --dataset <your-tts-dataset>

# Joint text + speech pre-training (helps retain language understanding)
# ratio=2 → 2 text batches followed by 1 speech batch, cycling
python pipeline.py --phase pretrain \
    --dataset <your-tts-dataset> \
    --text-dataset <your-qa-dataset> \
    --ratio 2
```

> **Why is there a separate `--text-dataset` for pre-training?**  
> The English Orpheus model was pre-trained with a mix of speech data and text QA data so it retains general language understanding (see `pretrain/readme.md`). For a new language the text dataset is **optional** — the default (`--ratio 0`) trains on speech only, which is the right choice when starting from the English base. Add `--text-dataset` only if you want to build or preserve strong text comprehension in the new language.

### Cache and reuse processed data

Encoding audio with SNAC can take a while for large datasets. Cache the result to HuggingFace and reuse it:

```bash
# First run: process and push
python pipeline.py --phase finetune --dataset <raw-dataset> --speaker zia \
    --push-processed-to <your-hf-username>/urdu-tts-processed

# Subsequent runs: skip processing entirely
python pipeline.py --phase finetune \
    --processed-dataset <your-hf-username>/urdu-tts-processed
```

### Multi-GPU training

Wrap any `pipeline.py` call with `accelerate launch` for distributed training:

```bash
accelerate launch pipeline.py --phase pretrain --dataset <your-tts-dataset>
```

---

## Finetune Model (manual / config-based)

Here is an overview of how to finetune your model on any text and speech.
This is a very simple process analogous to tuning an LLM using Trainer and Transformers.

You should start to see high quality results after ~50 examples but for best results, aim for 300 examples/speaker.

1. Your dataset should be a huggingface dataset in [this format](https://huggingface.co/datasets/canopylabs/zac-sample-dataset)
2. We prepare the data using [this notebook](https://colab.research.google.com/drive/1wg_CPCA-MzsWtsujwy-1Ovhv-tn8Q1nD?usp=sharing). This pushes an intermediate dataset to your Hugging Face account which you can can feed to the training script in finetune/train.py. Preprocessing should take less than 1 minute/thousand rows.
3. Modify the `finetune/config.yaml` file to include your dataset and training properties, and run the training script. You can additionally run any kind of huggingface compatible process like Lora to tune the model.
   ```bash
    pip install transformers datasets wandb trl flash_attn torch
    huggingface-cli login <enter your HF token>
    wandb login <wandb token>
    accelerate launch train.py
   ```

### Fine-tuning for Urdu (اردو)

#### Step 1 — Prepare your Urdu audio dataset

Collect Urdu speech recordings and their Nastaliq transcripts. Organise them as a HuggingFace dataset with the same schema used for English:

| Column | Type | Description |
|--------|------|-------------|
| `audio` | `Audio` | Raw audio file (any sample rate; will be resampled) |
| `text`  | `string` | Urdu transcript in Nastaliq script |
| `speaker` | `string` | Speaker identifier, e.g. `"zia"` |

You can push the raw dataset to your HuggingFace account:

```python
from datasets import Dataset, Audio
import pandas as pd

data = {
    "audio": ["path/to/clip1.wav", "path/to/clip2.wav"],  # paths to Urdu audio files
    "text":  ["یہ پہلا جملہ ہے۔", "یہ دوسرا جملہ ہے۔"],
    "speaker": ["zia", "zia"],
}
ds = Dataset.from_dict(data).cast_column("audio", Audio())
ds.push_to_hub("<YOUR_HF_USERNAME>/urdu-tts-raw")
```

Aim for at least **50 samples per speaker** for fine-tuning (300+ for best results). Pre-training requires substantially more data — typically thousands of hours of speech.

#### Step 2 — Preprocess into token IDs

Use the [data preprocessing notebook](https://colab.research.google.com/drive/1wg_CPCA-MzsWtsujwy-1Ovhv-tn8Q1nD?usp=sharing) to convert the raw audio+text pairs into model-ready `input_ids`. The notebook:
- Tokenises each Urdu text string with the `zia` voice prefix (`zia: <text>`)
- Encodes the audio using the SNAC codec into discrete tokens
- Concatenates text tokens and audio tokens into a single `input_ids` sequence
- Pushes the processed dataset to your HuggingFace account

#### Step 3 — Configure and run fine-tuning

Update `finetune/config.yaml` with the processed dataset path:

```yaml
# Urdu dataset (Nastaliq/Arabic script)
TTS_dataset: <YOUR_HF_USERNAME>/urdu-tts-processed
# language: "ur"

model_name: "canopylabs/orpheus-tts-0.1-pretrained"
```

Then run the training script (standard fine-tune):

```bash
pip install transformers datasets wandb trl flash_attn torch
huggingface-cli login
wandb login
cd finetune && accelerate launch train.py
```

Or use LoRA for parameter-efficient fine-tuning:

```bash
cd finetune && accelerate launch lora.py
```

During training every sample must use a consistent voice prefix that matches the speaker identifier in your dataset. `zia` is the recommended name for a default Urdu speaker, but you can use any name — simply keep it consistent between training data and inference:
```
zia: یہ ایک اردو جملہ ہے۔
```

At inference time, pass `voice="zia"` to `generate_speech()` when using a fine-tuned Urdu model. English voices (tara, leah, jess, etc.) remain fully supported and are unaffected by Urdu fine-tuning.
### Additional Resources
1. [Finetuning with unsloth](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb)
   
## Pretrain Model

This is a very simple process analogous to training an LLM using Trainer and Transformers.

The base model provided is trained over 100k hours. I recommend not using synthetic data for training as it produces worse results when you try to finetune specific voices, probably because synthetic voices lack diversity and map to the same set of tokens when tokenised (i.e. lead to poor codebook utilisation).

We train the 3b model on sequences of length 8192 - we use the same dataset format for TTS finetuning for the <TTS-dataset> pretraining. We chain input_ids sequences together for more efficient training. The text dataset required is in the form described in this issue [#37 ](https://github.com/canopyai/Orpheus-TTS/issues/37). 

If you are doing extended training this model, i.e. for another language or style we recommend starting with finetuning only (no text dataset). The main idea behind the text dataset is discussed in the blog post. (tldr; doesn't forget too much semantic/reasoning ability so its able to better understand how to intone/express phrases when spoken, however most of the forgetting would happen very early on in the training i.e. <100000 rows), so unless you are doing very extended finetuning it may not make too much of a difference.

### Pre-training for Urdu (اردو)

Pre-training from scratch (or continuing from the English base) on Urdu data lets the model learn Urdu phonetics and prosody at a deeper level than fine-tuning alone.

#### Step 1 — Prepare your Urdu speech dataset

Follow the same dataset schema as described in the Fine-tuning section above (audio + Nastaliq text + speaker columns). For pre-training you will typically need a **much larger** dataset — aim for thousands of hours of Urdu speech.

Push the raw dataset to HuggingFace and use the preprocessing notebook to convert it to `input_ids` sequences (see Fine-tuning Step 2).

#### Step 2 — (Optional) Prepare a tokenised Urdu text dataset

A text QA dataset in Urdu helps the model retain language-understanding ability during pre-training. Prepare a HuggingFace dataset of Urdu QA pairs tokenised with the Llama-3 tokeniser, following the format in [issue #37](https://github.com/canopyai/Orpheus-TTS/issues/37).

For Urdu-only TTS pre-training without a text dataset, set `ratio: 0` in `pretrain/config.yaml` — the `BatchedRatioDataset` will use only the speech dataset.

#### Step 3 — Configure pre-training

Update `pretrain/config.yaml`:

```yaml
model_name: "canopylabs/orpheus-tts-0.1-pretrained"  # continue from English base
tokenizer_name: "canopylabs/orpheus-tts-0.1-pretrained"

epochs: 1
batch_size: 1
number_processes: 8
pad_token: 128263
save_steps: 12000
learning_rate: 5.0e-5

# Set ratio to 0 for speech-only, or e.g. 2 to interleave 2 text batches per speech batch
ratio: 0

# Urdu datasets
text_QA_dataset: <YOUR_HF_USERNAME>/urdu-text-qa-tokenised   # omit / set ratio:0 if not used
TTS_dataset:     <YOUR_HF_USERNAME>/urdu-tts-processed

save_folder: "checkpoints"
project_name: "pretrain-orpheus-urdu"
run_name:     "urdu-pretrain-0"
```

#### Step 4 — Run pre-training

```bash
pip install transformers trl wandb flash_attn datasets torch
huggingface-cli login
wandb login
cd pretrain && accelerate launch train.py
```

After pre-training completes, the checkpoint in `checkpoints/` can be used directly as a base for Urdu fine-tuning (Step 3 of the Fine-tuning for Urdu guide above).

## Also Check out

While we can't verify these implementations are completely accurate/bug free, they have been recommended on a couple of forums, so we include them here:

1. [A lightweight client for running Orpheus TTS locally using LM Studio API](https://github.com/isaiahbjork/orpheus-tts-local)
2. [Open AI compatible Fast-API implementation](https://github.com/Lex-au/Orpheus-FastAPI)
3. [HuggingFace Space kindly set up by MohamedRashad](https://huggingface.co/spaces/MohamedRashad/Orpheus-TTS)
4. [Gradio WebUI that runs smoothly on WSL and CUDA](https://github.com/Saganaki22/OrpheusTTS-WebUI)


# Checklist

- [x] Release 3b pretrained model and finetuned models
- [ ] Release pretrained and finetuned models in sizes: 1b, 400m, 150m parameters
- [ ] Fix glitch in realtime streaming package that occasionally skips frames.
- [ ] Fix voice cloning Colab notebook implementation
