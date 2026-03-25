#!/usr/bin/env python3
"""
Orpheus TTS — single-entry-point training pipeline.

Handles the full workflow from raw audio/text data to a trained model:
  1. Loads a raw TTS dataset (audio + text columns, optional speaker column)
  2. Encodes audio with SNAC, tokenises text, builds ``input_ids`` sequences
  3. Launches fine-tuning (full or LoRA) **or** pre-training

For multi-GPU training, launch with ``accelerate``:
  accelerate launch pipeline.py --phase finetune --dataset <dataset>

-------------------------------------------------------------------------------
Usage examples
-------------------------------------------------------------------------------
Fine-tune on an English dataset (speaker column already present):
  python pipeline.py --phase finetune --dataset canopylabs/zac-sample-dataset

Fine-tune for Urdu (adds "zia" speaker prefix automatically):
  python pipeline.py --phase finetune --dataset <your-urdu-dataset> --speaker zia

Fine-tune for Urdu using LoRA (parameter-efficient):
  python pipeline.py --phase finetune --dataset <your-urdu-dataset> --speaker zia --lora

Pre-train (speech-only, no text QA dataset needed):
  python pipeline.py --phase pretrain --dataset <your-tts-dataset>

Pre-train with an interleaved text QA dataset (2 text batches per speech batch):
  python pipeline.py --phase pretrain --dataset <your-tts-dataset> \\
      --text-dataset <your-qa-dataset> --ratio 2

Skip raw processing if you already have a tokenised dataset:
  python pipeline.py --phase finetune --processed-dataset <already-tokenised-hf-repo>

-------------------------------------------------------------------------------
Why is the text QA dataset optional for pre-training?
-------------------------------------------------------------------------------
Interleaving a text QA dataset during pre-training helps the model retain
language-understanding ability (see pretrain/readme.md).  However, for most
Urdu / new-language scenarios – especially when starting from the Orpheus
English base – you only need the speech data (``--ratio 0``, the default).
Use ``--text-dataset`` only when you want the model to also learn / retain
strong text comprehension for the new language.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirror finetune/train.py and pretrain/train.py)
# ---------------------------------------------------------------------------
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"
DEFAULT_BASE_MODEL = "canopylabs/orpheus-tts-0.1-pretrained"

# Llama-3 base vocabulary size.  Orpheus custom tokens start here.
LLAMA_VOCAB_SIZE = 128256

# Special token IDs (custom_token_N → token_id = LLAMA_VOCAB_SIZE + N)
# <custom_token_3> — marks the start of a TTS prompt
START_TOKEN_ID = 128259
# <custom_token_2> — marks the end of the audio token sequence
AUDIO_END_TOKEN_ID = 128258
# Sequence that separates text from audio: eot_id + custom markers
END_TEXT_TOKEN_IDS = [128009, 128260, 128261, 128257]
# Padding token — <custom_token_7>
PAD_TOKEN_ID = 128263

# Number of custom tokens to register (7 codebook levels × 4096 + 11 overhead)
NUM_CUSTOM_TOKENS = 7 * 4096 + 11

TARGET_SAMPLE_RATE = 24_000


# ---------------------------------------------------------------------------
# Audio encoding
# ---------------------------------------------------------------------------

def _load_snac():
    import torch
    from snac import SNAC  # pip install snac
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SNAC.from_pretrained(SNAC_MODEL_ID).eval().to(device)
    return model, device


def _resample(waveform, src_sr: int):
    import torchaudio  # pip install torchaudio
    if src_sr != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, src_sr, TARGET_SAMPLE_RATE)
    return waveform


def encode_audio_to_token_ids(
    waveform,
    sample_rate: int,
    snac_model,
    device: str,
) -> list[int]:
    """Encode a waveform to a flat list of interleaved SNAC token IDs.

    The SNAC codec produces three codebook levels:
      codes[0]: shape [1, T]       — 1 code  per frame
      codes[1]: shape [1, 2T]      — 2 codes per frame
      codes[2]: shape [1, 4T]      — 4 codes per frame

    Each frame is represented as 7 consecutive token IDs in the order:
      [c0_j, c1_2j, c2_4j, c2_4j+1, c1_2j+1, c2_4j+2, c2_4j+3]

    Token ID formula:
      token_id = LLAMA_VOCAB_SIZE + codebook_value + 10 + (position_in_frame % 7) * 4096
    """
    import torch

    waveform = _resample(waveform, sample_rate)
    # Ensure mono float tensor with shape [1, samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.to(device)

    with torch.inference_mode():
        codes = snac_model.encode(waveform.unsqueeze(0))

    c0 = codes[0][0].tolist()   # length T
    c1 = codes[1][0].tolist()   # length 2T
    c2 = codes[2][0].tolist()   # length 4T
    T = len(c0)

    token_ids: list[int] = []
    offset = LLAMA_VOCAB_SIZE + 10
    for j in range(T):
        token_ids.append(offset + c0[j]           + 0 * 4096)
        token_ids.append(offset + c1[2 * j]       + 1 * 4096)
        token_ids.append(offset + c2[4 * j]       + 2 * 4096)
        token_ids.append(offset + c2[4 * j + 1]   + 3 * 4096)
        token_ids.append(offset + c1[2 * j + 1]   + 4 * 4096)
        token_ids.append(offset + c2[4 * j + 2]   + 5 * 4096)
        token_ids.append(offset + c2[4 * j + 3]   + 6 * 4096)
    return token_ids


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------

def build_input_ids(
    text: str,
    speaker: str,
    audio_token_ids: list[int],
    tokenizer,
) -> list[int]:
    """Assemble a complete training sequence for one audio+text pair.

    Format:
      [START] + tokenise("{speaker}: {text}") + [END_TEXT...] + [audio tokens] + [AUDIO_END]
    """
    text_with_voice = f"{speaker}: {text}"
    text_ids = tokenizer(text_with_voice, add_special_tokens=False).input_ids
    return (
        [START_TOKEN_ID]
        + text_ids
        + END_TEXT_TOKEN_IDS
        + audio_token_ids
        + [AUDIO_END_TOKEN_ID]
    )


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def process_dataset(
    raw_path: str,
    speaker: Optional[str],
    model_name: str,
    hf_push_repo: Optional[str] = None,
    max_samples: Optional[int] = None,
):
    """Convert a raw audio+text dataset into tokenised ``input_ids`` sequences.

    Parameters
    ----------
    raw_path:
        HuggingFace dataset repo ID **or** local folder accepted by
        ``load_dataset``.  The dataset must have:
          - ``audio``  column (``datasets.Audio``)
          - ``text``   column (or ``transcript``)
          - ``speaker`` column — optional; if absent, ``speaker`` arg must be set.
    speaker:
        Voice/speaker prefix (e.g. ``"tara"``, ``"zia"``).  If ``None``, the
        value is read from the dataset's ``speaker`` column for every row.
        If neither is available, a ``ValueError`` is raised.
    model_name:
        Base model used to load the tokenizer.
    hf_push_repo:
        If set, the processed dataset is pushed to this HuggingFace repo.
    max_samples:
        Limit number of samples processed (useful for smoke-tests).

    Returns
    -------
    datasets.Dataset
        Dataset with a single ``input_ids`` column (list of int).
    """
    from datasets import load_dataset, Audio as HFAudio
    from transformers import AutoTokenizer

    log.info("Loading tokenizer from %s …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    custom_tokens = [f"<custom_token_{i}>" for i in range(NUM_CUSTOM_TOKENS)]
    tokenizer.add_tokens(custom_tokens)

    log.info("Loading SNAC encoder (%s) …", SNAC_MODEL_ID)
    snac_model, device = _load_snac()

    log.info("Loading raw dataset from %s …", raw_path)
    try:
        raw_ds = load_dataset(raw_path, split="train")
    except Exception as exc:
        log.warning("Could not load as a HuggingFace repo (%s); trying as local parquet …", exc)
        raw_ds = load_dataset("parquet", data_dir=raw_path, split="train")

    if max_samples:
        raw_ds = raw_ds.select(range(min(max_samples, len(raw_ds))))

    # Ensure audio is decoded and resampled to TARGET_SAMPLE_RATE
    if "audio" in raw_ds.column_names:
        raw_ds = raw_ds.cast_column("audio", HFAudio(sampling_rate=TARGET_SAMPLE_RATE))
    else:
        raise ValueError("Dataset must have an 'audio' column.")

    has_speaker_col = "speaker" in raw_ds.column_names

    if speaker is None and not has_speaker_col:
        raise ValueError(
            "Dataset has no 'speaker' column and --speaker was not provided.  "
            "Pass --speaker <voice_name> (e.g. --speaker zia) to assign a speaker "
            "prefix for all rows."
        )

    if speaker is not None and not has_speaker_col:
        log.info("No 'speaker' column found — using '%s' for all rows.", speaker)
    elif speaker is not None and has_speaker_col:
        log.info(
            "Both --speaker and a 'speaker' column are present; "
            "--speaker '%s' will override the column value.",
            speaker,
        )

    def _process_row(row):
        import torch
        # Resolve speaker name
        row_speaker = speaker if speaker is not None else row["speaker"]

        # Audio
        audio_data = row["audio"]
        waveform = torch.tensor(audio_data["array"], dtype=torch.float32).unsqueeze(0)
        sr = audio_data["sampling_rate"]

        # Text
        if "text" in row:
            text = row["text"]
        elif "transcript" in row:
            text = row["transcript"]
        else:
            raise ValueError("Dataset must have a 'text' or 'transcript' column.")

        audio_ids = encode_audio_to_token_ids(waveform, sr, snac_model, device)
        input_ids = build_input_ids(text, row_speaker, audio_ids, tokenizer)
        return {"input_ids": input_ids}

    log.info(
        "Processing %d samples (SNAC encoding + tokenisation) …", len(raw_ds)
    )
    processed = raw_ds.map(
        _process_row,
        remove_columns=raw_ds.column_names,
        desc="Encoding audio & tokenising",
    )

    if hf_push_repo:
        log.info("Pushing processed dataset to %s …", hf_push_repo)
        processed.push_to_hub(hf_push_repo)
        log.info("Dataset pushed.  Pass --processed-dataset %s to skip re-processing.", hf_push_repo)

    return processed


# ---------------------------------------------------------------------------
# Data collator (shared by both fine-tune and pretrain)
# ---------------------------------------------------------------------------

def _make_data_collator(pad_id: int):
    def _collate(features):
        import torch
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = [
            torch.tensor(f.get("labels", f["input_ids"]), dtype=torch.long)
            for f in features
        ]
        attn_mask = [torch.ones(len(f["input_ids"]), dtype=torch.long) for f in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attn_mask = torch.nn.utils.rnn.pad_sequence(
            attn_mask, batch_first=True, padding_value=0
        )
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

    return _collate


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(model_name: str, lora: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    custom_tokens = [f"<custom_token_{i}>" for i in range(NUM_CUSTOM_TOKENS)]
    tokenizer.add_tokens(custom_tokens)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="flash_attention_2"
    )
    model.resize_token_embeddings(len(tokenizer))

    if lora:
        from peft import LoraConfig, get_peft_model  # pip install peft

        lora_cfg = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "down_proj", "up_proj",
            ],
            bias="none",
            modules_to_save=["lm_head", "embed_tokens"],
            task_type="CAUSAL_LM",
            use_rslora=True,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def run_finetune(processed_dataset, args):
    """Full fine-tune or LoRA fine-tune on a processed TTS dataset."""
    import wandb
    from transformers import Trainer, TrainingArguments

    wandb.init(project="tuning-orpheus", name=args.run_name or "finetune-run")

    model, _ = _load_model_and_tokenizer(args.model, lora=args.lora)

    training_args = TrainingArguments(
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_steps=1,
        bf16=True,
        output_dir=args.output_dir,
        report_to="wandb",
        save_steps=args.save_steps,
        remove_unused_columns=False,
        learning_rate=args.lr,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=_make_data_collator(PAD_TOKEN_ID),
    )
    trainer.train()

    if args.lora:
        log.info("Merging LoRA weights …")
        merged = model.merge_and_unload()
        merged_dir = os.path.join(args.output_dir, "merged")
        merged.save_pretrained(merged_dir)
        log.info("Merged model saved to %s", merged_dir)


# ---------------------------------------------------------------------------
# Pre-training (with optional interleaved text QA dataset)
# ---------------------------------------------------------------------------

def run_pretrain(processed_tts_dataset, text_qa_dataset, args):
    """Pre-training with an optional interleaved text QA dataset.

    When ``text_qa_dataset`` is ``None`` (or ``--ratio 0``), only the speech
    dataset is used.  This is the recommended starting point for a new language
    like Urdu when continuing from the English Orpheus base model.
    """
    import torch
    import wandb
    from transformers import Trainer, TrainingArguments

    # Define dataset class here so torch is imported lazily
    class _BatchedRatioDataset(torch.utils.data.Dataset):
        """Interleaves two datasets at a fixed ratio.

        ``ratio`` batches from ``text_dataset`` are followed by 1 batch from
        ``speech_dataset``.  Use ``ratio=0`` to train on speech only.
        """

        def __init__(self, text_dataset, speech_dataset, batch_total: int, ratio: int):
            self.ds_text = text_dataset
            self.ds_speech = speech_dataset
            self.batch_total = batch_total
            self.ratio = ratio

            if ratio == 0:
                self.length = len(speech_dataset)
            else:
                # One cycle = (ratio text batches + 1 speech batch) × batch_total samples.
                # Total length is capped at the smaller of the two datasets' capacities.
                n_cycles_text = len(text_dataset) // (batch_total * ratio)
                n_cycles_speech = len(speech_dataset) // batch_total
                n_cycles = min(n_cycles_text, n_cycles_speech)
                self.length = n_cycles * (ratio + 1) * batch_total

        def __len__(self):
            return int(self.length)

        def __getitem__(self, index):
            if self.ratio == 0:
                return self.ds_speech[index]

            # Determine which cycle this index falls in, and where within it.
            cycle_len = (self.ratio + 1) * self.batch_total
            cycle = index // cycle_len
            pos = index % cycle_len

            if pos < self.ratio * self.batch_total:
                # First (ratio × batch_total) positions in the cycle → text dataset
                batch_in_cycle = pos // self.batch_total
                sample_in_batch = pos % self.batch_total
                ds_idx = (
                    cycle * self.ratio * self.batch_total
                    + batch_in_cycle * self.batch_total
                    + sample_in_batch
                )
                return self.ds_text[ds_idx]
            else:
                # Last batch_total positions in the cycle → speech dataset
                sample_in_batch = pos - self.ratio * self.batch_total
                ds_idx = cycle * self.batch_total + sample_in_batch
                return self.ds_speech[ds_idx]

    wandb.init(project="pretrain-orpheus", name=args.run_name or "pretrain-run")

    model, _ = _load_model_and_tokenizer(args.model)

    ratio = args.ratio if (text_qa_dataset is not None and args.ratio > 0) else 0

    if ratio == 0 or text_qa_dataset is None:
        log.info("Pre-training on speech data only (ratio=0).")
        train_dataset = processed_tts_dataset
    else:
        log.info(
            "Pre-training with interleaved text QA (ratio=%d text : 1 speech).", ratio
        )
        batch_total = args.batch_size
        train_dataset = _BatchedRatioDataset(
            text_qa_dataset, processed_tts_dataset, batch_total, ratio
        )

    training_args = TrainingArguments(
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_steps=1,
        bf16=True,
        output_dir=args.output_dir,
        fsdp="auto_wrap",
        report_to="wandb",
        save_steps=args.save_steps,
        remove_unused_columns=False,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=_make_data_collator(PAD_TOKEN_ID),
    )
    trainer.train()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Orpheus TTS Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--phase",
        required=True,
        choices=["finetune", "pretrain"],
        help="Training phase: 'finetune' (from base) or 'pretrain' (from scratch / base).",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help=(
            "Raw TTS dataset — HuggingFace repo ID or local path.  "
            "Must have 'audio' and 'text' (or 'transcript') columns.  "
            "Ignored when --processed-dataset is provided."
        ),
    )
    p.add_argument(
        "--speaker",
        default=None,
        help=(
            "Speaker / voice prefix (e.g. 'tara', 'zia').  "
            "If the dataset already has a 'speaker' column this flag overrides it.  "
            "Required when the dataset has no 'speaker' column."
        ),
    )
    p.add_argument(
        "--model",
        default=DEFAULT_BASE_MODEL,
        help=f"Base model for training (default: {DEFAULT_BASE_MODEL}).",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5, metavar="LEARNING_RATE")
    p.add_argument("--save-steps", type=int, default=5000)
    p.add_argument(
        "--output-dir",
        default="checkpoints",
        help="Local directory to save model checkpoints.",
    )
    p.add_argument(
        "--lora",
        action="store_true",
        help="[finetune only] Use LoRA for parameter-efficient fine-tuning.",
    )
    p.add_argument(
        "--text-dataset",
        default=None,
        metavar="HF_REPO_OR_PATH",
        help=(
            "[pretrain only] Optional text QA dataset for joint text+speech training.  "
            "When omitted (default), pre-training uses speech data only (ratio=0).  "
            "See pretrain/readme.md for details on why this can help."
        ),
    )
    p.add_argument(
        "--ratio",
        type=int,
        default=0,
        help=(
            "[pretrain only] Number of text QA batches per speech batch.  "
            "0 = speech-only (default).  Ignored when --text-dataset is not set."
        ),
    )
    p.add_argument(
        "--push-processed-to",
        default=None,
        metavar="HF_REPO",
        help=(
            "After encoding, push the processed dataset to this HuggingFace repo.  "
            "Useful to cache the result and reuse it with --processed-dataset."
        ),
    )
    p.add_argument(
        "--processed-dataset",
        default=None,
        metavar="HF_REPO_OR_PATH",
        help=(
            "Skip raw audio processing and use an already-tokenised dataset "
            "(HuggingFace repo ID or local path with an 'input_ids' column)."
        ),
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of processed samples — useful for quick smoke-tests.",
    )
    p.add_argument("--run-name", default=None, help="Weights & Biases run name.")

    return p.parse_args()


def main():
    args = parse_args()

    if args.processed_dataset is None and args.dataset is None:
        raise SystemExit("Either --dataset or --processed-dataset must be provided.")

    # ── Step 1: Data processing ──────────────────────────────────────────────
    if args.processed_dataset:
        from datasets import load_dataset

        log.info("Loading pre-processed dataset from %s …", args.processed_dataset)
        try:
            processed = load_dataset(args.processed_dataset, split="train")
        except Exception as exc:
            log.warning("Could not load as a HuggingFace repo (%s); trying as local parquet …", exc)
            processed = load_dataset("parquet", data_dir=args.processed_dataset, split="train")
    else:
        processed = process_dataset(
            raw_path=args.dataset,
            speaker=args.speaker,
            model_name=args.model,
            hf_push_repo=args.push_processed_to,
            max_samples=args.max_samples,
        )

    # ── Step 2: Training ────────────────────────────────────────────────────
    if args.phase == "finetune":
        log.info("Starting fine-tuning (lora=%s) …", args.lora)
        run_finetune(processed, args)

    elif args.phase == "pretrain":
        text_ds = None
        if args.text_dataset:
            from datasets import load_dataset

            log.info("Loading text QA dataset from %s …", args.text_dataset)
            try:
                text_ds = load_dataset(args.text_dataset, split="train")
            except Exception as exc:
                log.warning("Could not load as a HuggingFace repo (%s); trying as local parquet …", exc)
                text_ds = load_dataset("parquet", data_dir=args.text_dataset, split="train")

        effective_ratio = args.ratio if text_ds is not None else 0
        log.info(
            "Starting pre-training (ratio=%d, text_dataset=%s) …",
            effective_ratio,
            args.text_dataset or "none",
        )
        run_pretrain(processed, text_ds, args)


if __name__ == "__main__":
    main()
