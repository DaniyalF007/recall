import time
from PIL import Image
from pdf2image import convert_from_path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def load_trocr():
    """Load TrOCR processor and model."""
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-printed")
    return processor, model


def ocr_image(image: Image.Image, processor, model) -> str:
    """Extract text from a single image using TrOCR."""
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


def extract_text_from_scanned_pdf(file_path: str) -> tuple[str, float]:
    """Convert scanned PDF pages to images and extract text using TrOCR."""
    start = time.time()
    processor, model = load_trocr()
    images = convert_from_path(file_path)
    full_text = ""
    for image in images:
        page_text = ocr_image(image, processor, model)
        full_text += page_text + "\n"
    latency = time.time() - start
    return full_text, latency
