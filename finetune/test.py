from pathlib import Path
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import logging
import sys
import traceback
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).parent / "finetuned" / "best-model"


def load_model(model_path):
    """Load processor and model from model_path.

    Returns:
        tuple: (processor, model, device)
    """
    model_path = str(model_path)
    logger.info(f"Loading model from {model_path}...")
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    model = model.to(device)
    model.eval()
    return processor, model, device


def run_ocr(image_path, processor, model: VisionEncoderDecoderModel, device):
    """Run OCR on an image with preloaded processor/model.

    Args:
        image_path: Path to the image file to OCR
        processor: Loaded TrOCRProcessor
        model: Loaded VisionEncoderDecoderModel
        device: torch.device

    Returns:
        str: Recognized text
    """
    logger.info(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=7,
            num_beams=4,
        )

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def test_ocr(image_path, model_path=None):
    """Convenience wrapper: load model then run OCR.

    Args:
        image_path: Path to the image file to OCR
        model_path: Path to the trained model (default: DEFAULT_MODEL_PATH)
    """
    processor, model, device = load_model(model_path or DEFAULT_MODEL_PATH)
    return run_ocr(image_path, processor, model, device)


def test_folder(folder_path, processor, model, device, folder_name):
    """Test all images in a specific folder.
    
    Args:
        folder_path: Path to folder containing images
        processor: Loaded TrOCRProcessor
        model: Loaded VisionEncoderDecoderModel
        device: torch.device
        folder_name: Name of the folder (for logging)
    
    Returns:
        tuple: (total_images, correct_predictions, mismatches)
    """
    if not folder_path.exists():
        logger.warning(f"{folder_name} folder does not exist: {folder_path}")
        return 0, 0, []
    
    image_files = sorted(folder_path.glob("*.png"))
    
    if not image_files:
        logger.warning(f"No PNG files found in {folder_path}")
        return 0, 0, []
    
    logger.info(f"\nTesting {len(image_files)} images in {folder_name} folder...")
    
    correct = 0
    mismatches = []
    
    for image_path in image_files:
        expected_text = image_path.stem  # Filename without extension
        result = run_ocr(image_path, processor, model, device)[:5]
        
        if result == expected_text:
            logger.info(f"✓ {image_path.name}: {result}")
            correct += 1
        else:
            logger.warning(f"✗ {image_path.name}: expected '{expected_text}', got '{result}'")
            mismatches.append({
                "file": image_path.name,
                "expected": expected_text,
                "got": result
            })
    
    return len(image_files), correct, mismatches


if __name__ == "__main__":
    test_dir = Path(__file__).parent.parent / "data" / "CaptchaTest"

    if not test_dir.exists():
        logger.error(f"Test directory does not exist: {test_dir.resolve()}")
        sys.exit(1)

    try:
        # Load model once
        processor, model, device = load_model(DEFAULT_MODEL_PATH)

        # Test CaptchaTest folder
        total_images, total_correct, total_mismatches = test_folder(
            test_dir, processor, model, device, "CaptchaTest"
        )

        # Print summary
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)

        if total_images > 0:
            overall_accuracy = (total_correct / total_images) * 100
            print(f"\nTotal:    {total_images} images")
            print(f"Correct:  {total_correct}")
            print(f"Accuracy: {overall_accuracy:.2f}%")
        
        # Show mismatches if any
        if total_mismatches:
            print("\n" + "=" * 60)
            print("MISMATCHES DETECTED:")
            print("=" * 60)
            for mismatch in total_mismatches:
                print(f"\n❌ File: {mismatch['file']}")
                print(f"   Expected: {mismatch['expected']}")
                print(f"   Got:      {mismatch['got']}")
            print("=" * 60)
            sys.exit(1)
        else:
            print("\n" + "=" * 60)
            print("✓ ALL TESTS PASSED!")
            print("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception:
        logger.error("Unknown error:\n" + traceback.format_exc())
        sys.exit(1)
