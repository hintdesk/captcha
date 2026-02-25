from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    EarlyStoppingCallback,
)
from datasets import load_dataset
from pathlib import Path
import torch
import logging
import json
from datetime import datetime
import time
import random
import numpy as np
from PIL import Image, ImageFilter
from jiwer import cer
from functools import partial

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

hugginface_model_name = "microsoft/trocr-base-printed"

def compute_metrics(pred, processor):
    """Compute Character Error Rate (CER) for model predictions."""
    
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    # Replace -100 in the predictions and labels with pad_token_id for decoding
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    
    # Calculate CER
    cer_value = cer(label_str, pred_str)
    print(f"Sample Label: {label_str[0]} | Sample Pred: {pred_str[0]}")
    return {"cer": cer_value}


def augment(img):
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, fillcolor=255, expand=False)

    if random.random() < 0.5:
        dx = random.randint(-8, 8)
        dy = random.randint(-4, 4)
        img = img.transform(
            img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=255
        )

    if random.random() < 0.3:
        w, h = img.size
        scale = random.uniform(0.85, 1.15)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        result = Image.new("L", (w, h), 255)
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2
        result.paste(img, (paste_x, paste_y))
        img = result

    if random.random() < 0.4:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, 15, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    if random.random() < 0.3:
        if random.random() < 0.5:
            img = img.filter(ImageFilter.MinFilter(3))
        else:
            img = img.filter(ImageFilter.MaxFilter(3))

    if random.random() < 0.4:
        arr = np.array(img, dtype=np.float32)
        alpha = random.uniform(0.7, 1.3)
        beta = random.uniform(-30, 30)
        arr = np.clip(alpha * arr + beta, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def preprocess_batch(examples, processor, do_augment):
    images = []
    for image in examples["image"]:
        img = image.convert("L")
        if do_augment:
            img = augment(img)
        images.append(img.convert("RGB"))

    pixel_values = processor(images=images, return_tensors="pt").pixel_values

    # Encode labels with padding
    encoded = processor.tokenizer(
        examples["text"], padding="max_length", max_length=20, truncation=True
    ).input_ids

    # Set padding tokens to -100 so trainer ignores them in loss
    pad_id = processor.tokenizer.pad_token_id
    labels = [[(token if token != pad_id else -100) for token in seq] for seq in encoded]

    return {"pixel_values": pixel_values, "labels": labels}

def main():
    start_time = time.time()
    logger.info("Loading processor...")
    processor = TrOCRProcessor.from_pretrained(hugginface_model_name)
    logger.info("Loading model...")
    model = VisionEncoderDecoderModel.from_pretrained(hugginface_model_name)

    # Set pad_token_id on model config
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    model = model.to(device)

    logger.info("Loading dataset...")
    data_dir = Path(__file__).parent.parent / "data" / "CaptchaDatasets"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir.resolve()}")
    dataset = load_dataset("imagefolder", data_dir=str(data_dir))
    logger.info(f"Dataset loaded: {dataset}")

    train_transform = partial(preprocess_batch, processor=processor, do_augment=True)
    val_transform = partial(preprocess_batch, processor=processor, do_augment=False)

    # Dataset automatically loaded from train/ and val/ folders

    logger.info(
        f"Train size: {len(dataset['train'])}, Validation size: {len(dataset['validation'])}"
    )
    logger.info("Setting dataset transforms (augmentation on train split)...")
    dataset["train"].set_transform(train_transform)
    dataset["validation"].set_transform(val_transform)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./finetuned",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=30,
        predict_with_generate=True,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        logging_steps=10,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")

    # Save best model and processor
    logger.info("Saving model and processor...")
    trainer.save_model("./finetuned/best-model")
    processor.save_pretrained("./finetuned/best-model")
    logger.info(f"Model saved at: {Path('./finetuned/best-model').resolve()}")

    # Calculate total execution time
    total_time = time.time() - start_time

    # Save training history to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path("./finetuned") / f"training_log_{timestamp}.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    training_log = {
        "timestamp": timestamp,
        "batch_size": training_args.per_device_train_batch_size,
        "num_train_epochs": training_args.num_train_epochs,
        "dataloader_num_workers": training_args.dataloader_num_workers,
        "learning_rate": float(training_args.learning_rate),
        "total_time_seconds": total_time,
        "log_history": trainer.state.log_history,
    }

    with open(log_file, "w") as f:
        json.dump(training_log, f, indent=2)
    logger.info(f"Training log saved to: {log_file.resolve()}")
    logger.info(
        f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )


if __name__ == "__main__":
    main()
