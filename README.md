# CAPTCHA Solver Project

This project focuses on solving CAPTCHA recognition.

## Dataset Layout

- [CaptchaRaw/](CaptchaRaw/): source data for **training** and **validation**.
- [CaptchaTest/](CaptchaTest/): source data for **testing only**.

## Data Split Rule (Important)

Files in [CaptchaTest/](CaptchaTest/) must never be used during training or validation.

Use [CaptchaRaw/](CaptchaRaw/) only to build train/validation datasets, then evaluate final performance on [CaptchaTest/](CaptchaTest/).

## Available Solutions

### 1) Multihead CNN

- Implementation folder: [multiheadcnn/](multiheadcnn/)
- Accuracy: **100%**
- Hardware: **No GPU required**
- Training time: **~10 minutes**

### 2) Fine-tune

- Implementation folder: [finetune/](finetune/)
- Accuracy: **~93%**
- Hardware: **GPU required**
- Training time: **~90 minutes**

## Quick Comparison

| Method | Accuracy | GPU | Training Time |
|---|---:|---|---:|
| Multihead CNN | 100% | Not required | ~10 min |
| Fine-tune | 93% | Required | ~90 min |

## Credits

- Multihead CNN approach credited to [neoz/captcha_finetune](https://github.com/neoz/captcha_finetune).
- Repository owner: [@neoz](https://github.com/neoz).
