import json
import argparse
import os

# Import the functions from your modules
from src.ocr.vision_ocr import ocr_image
from src.preprocess.normalize import group_words_into_lines, apply_common_fixes
from src.parser.rule_parser import extract_fields_by_rules

def run_pipeline(image_path: str):
    """
    Executes the full OCR and extraction pipeline for a single image.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # Define paths for intermediate files
    filename_base = os.path.splitext(os.path.basename(image_path))[0]
    ocr_output_path = f"data/ocr_jsons/{filename_base}.json"

    # 1. Run OCR
    ocr_image(image_path, ocr_output_path)

    # 2. Load the OCR result
    with open(ocr_output_path, "r", encoding='utf-8') as f:
        ocr_data = json.load(f)
    
    words = ocr_data.get("words", [])

    # 3. Preprocess: Group words into lines
    lines = group_words_into_lines(words)

    # 4. Parse: Extract fields using rules
    extracted_data = extract_fields_by_rules(lines)
    
    # Clean the extracted values
    for key, value in extracted_data.items():
        extracted_data[key] = apply_common_fixes(value)

    print("\n--- EXTRACTION COMPLETE ---")
    print(json.dumps(extracted_data, indent=2))
    print("---------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full OCR extraction pipeline.")
    parser.add_argument(
        "--image-path",
        required=True,
        help="Path to the input image file."
    )
    args = parser.parse_args()
    
    run_pipeline(args.image_path)