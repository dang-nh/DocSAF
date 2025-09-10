#!/usr/bin/env python3

import os
import json
import shutil
import logging
import argparse
from pathlib import Path
import zipfile
from PIL import Image


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def process_funsd(data_dir, extract_archives=False):
    """Process FUNSD dataset to ensure it has the correct structure.
    
    Args:
        data_dir: Base data directory.
        extract_archives: Whether to extract from archives if needed
    """
    funsd_dir = Path(data_dir) / "funsd"
    archives_dir = Path(data_dir) / "archives"
    funsd_archive = archives_dir / "FUNSD.zip"
    
    # Create directory structure if it doesn't exist
    funsd_dir.mkdir(exist_ok=True)
    training_data_dir = funsd_dir / "training_data"
    testing_data_dir = funsd_dir / "testing_data"
    
    # Check if data already exists
    if (training_data_dir / "images").exists() and (testing_data_dir / "images").exists():
        logging.info("FUNSD dataset already properly organized.")
        return
    
    # Extract from archive if requested
    if extract_archives and funsd_archive.exists():
        logging.info(f"Extracting FUNSD dataset from {funsd_archive}")
        extract_dir = funsd_dir / "temp_extract"
        
        with zipfile.ZipFile(funsd_archive, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # FUNSD typically has a dataset/ folder with training_data/ and testing_data/
        extracted_dataset_dir = extract_dir / "dataset"
        
        # Move files to the correct locations
        if extracted_dataset_dir.exists():
            if (extracted_dataset_dir / "training_data").exists():
                logging.info("Moving training data to final location")
                if training_data_dir.exists():
                    shutil.rmtree(training_data_dir)
                shutil.move(str(extracted_dataset_dir / "training_data"), str(training_data_dir))
            
            if (extracted_dataset_dir / "testing_data").exists():
                logging.info("Moving testing data to final location")
                if testing_data_dir.exists():
                    shutil.rmtree(testing_data_dir)
                shutil.move(str(extracted_dataset_dir / "testing_data"), str(testing_data_dir))
        
        # Clean up
        shutil.rmtree(extract_dir)
    else:
        if not funsd_archive.exists() and extract_archives:
            logging.error(f"Archive {funsd_archive} not found.")
        if not (training_data_dir / "images").exists() or not (testing_data_dir / "images").exists():
            logging.error("FUNSD dataset not properly organized and no archive to extract.")
            return
    
    # Validate dataset structure
    if not validate_funsd_structure(funsd_dir):
        return
    
    # Process annotations to ensure consistent format
    process_funsd_annotations(training_data_dir)
    process_funsd_annotations(testing_data_dir)
    
    logging.info("FUNSD dataset processing completed.")


def validate_funsd_structure(funsd_dir):
    """Validate that the FUNSD dataset has the correct directory structure."""
    training_images = funsd_dir / "training_data" / "images"
    training_annotations = funsd_dir / "training_data" / "annotations"
    testing_images = funsd_dir / "testing_data" / "images"
    testing_annotations = funsd_dir / "testing_data" / "annotations"
    
    valid = True
    if not training_images.exists():
        logging.error(f"Missing directory: {training_images}")
        valid = False
    if not training_annotations.exists():
        logging.error(f"Missing directory: {training_annotations}")
        valid = False
    if not testing_images.exists():
        logging.error(f"Missing directory: {testing_images}")
        valid = False
    if not testing_annotations.exists():
        logging.error(f"Missing directory: {testing_annotations}")
        valid = False
    
    if valid:
        logging.info("FUNSD directory structure validation successful.")
    else:
        logging.error("FUNSD directory structure validation failed.")
    
    return valid


def process_funsd_annotations(data_dir):
    """Process FUNSD annotations to ensure they're in the correct format."""
    annotations_dir = data_dir / "annotations"
    images_dir = data_dir / "images"
    
    for annotation_file in annotations_dir.glob("*.json"):
        with open(annotation_file, "r", encoding="utf-8") as f:
            annotation = json.load(f)
        
        # Ensure annotation has the expected structure
        if "form" not in annotation:
            logging.error(f"Invalid annotation format in {annotation_file}")
            continue
        
        # Check if corresponding image exists
        image_file = images_dir / annotation_file.name.replace(".json", ".png")
        if not image_file.exists():
            logging.warning(f"Image file not found: {image_file}")
            continue
        
        # Validate and fix bounding boxes if needed
        modified = False
        for item in annotation["form"]:
            for word in item["words"]:
                box = word["box"]
                # Ensure box has [x1, y1, x2, y2] format and values are integers
                if len(box) != 4:
                    logging.warning(f"Invalid box format in {annotation_file}: {box}")
                    continue
                    
                # Convert box values to integers if they're not already
                if not all(isinstance(coord, int) for coord in box):
                    word["box"] = [int(coord) for coord in box]
                    modified = True
                    
                # Ensure x1 <= x2 and y1 <= y2
                x1, y1, x2, y2 = word["box"]
                if x1 > x2 or y1 > y2:
                    word["box"] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    modified = True
                
                # Ensure bounding box is not out of image bounds
                try:
                    img = Image.open(image_file)
                    width, height = img.size
                    x1, y1, x2, y2 = word["box"]
                    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                        word["box"] = [
                            max(0, x1),
                            max(0, y1),
                            min(width, x2),
                            min(height, y2)
                        ]
                        modified = True
                except Exception as e:
                    logging.error(f"Error processing image {image_file}: {e}")
        
        # Save modified annotation if changes were made
        if modified:
            with open(annotation_file, "w", encoding="utf-8") as f:
                json.dump(annotation, f, indent=2)
    
    logging.info(f"Processed annotations in {data_dir}")


def process_sroie(data_dir, extract_archives=False):
    """Process SROIE dataset to ensure it has the correct structure."""
    sroie_dir = Path(data_dir) / "sroie"
    archives_dir = Path(data_dir) / "archives"
    sroie_archive = archives_dir / "sroie.zip"
    
    # Create directory structure if it doesn't exist
    sroie_dir.mkdir(exist_ok=True)
    train_dir = sroie_dir / "train"
    test_dir = sroie_dir / "test"
    
    for split_dir in [train_dir, test_dir]:
        split_dir.mkdir(exist_ok=True)
        (split_dir / "images").mkdir(exist_ok=True)
        (split_dir / "tagged").mkdir(exist_ok=True)
        (split_dir / "images_draw").mkdir(exist_ok=True)
    
    # Check if data already exists
    if len(list((train_dir / "images").glob("*.jpg"))) > 0 and len(list((test_dir / "images").glob("*.jpg"))) > 0:
        logging.info("SROIE dataset already properly organized.")
        return
    
    # Extract from archive if requested
    if extract_archives and sroie_archive.exists():
        logging.info(f"Extracting SROIE dataset from {sroie_archive}")
        extract_dir = sroie_dir / "temp_extract"
        
        with zipfile.ZipFile(sroie_archive, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Process the extracted data - structure varies by source
        # This is a placeholder - adjust based on actual archive structure
        for extracted_file in extract_dir.glob("**/*.jpg"):
            if "train" in str(extracted_file).lower():
                shutil.copy(extracted_file, train_dir / "images")
            elif "test" in str(extracted_file).lower():
                shutil.copy(extracted_file, test_dir / "images")
        
        for extracted_file in extract_dir.glob("**/*.txt"):
            if "train" in str(extracted_file).lower():
                shutil.copy(extracted_file, train_dir / "tagged")
            elif "test" in str(extracted_file).lower():
                shutil.copy(extracted_file, test_dir / "tagged")
        
        # Clean up
        shutil.rmtree(extract_dir)
    else:
        if not sroie_archive.exists() and extract_archives:
            logging.error(f"Archive {sroie_archive} not found.")
        if len(list((train_dir / "images").glob("*.jpg"))) == 0 or len(list((test_dir / "images").glob("*.jpg"))) == 0:
            logging.error("SROIE dataset not properly organized and no archive to extract.")
            return
    
    logging.info("SROIE dataset processing completed.")


def process_cord(data_dir, extract_archives=False):
    """Process CORD dataset to ensure it has the correct structure."""
    cord_dir = Path(data_dir) / "cord-v2"
    archives_dir = Path(data_dir) / "archives"
    cord_archive = archives_dir / "CORD-v2.zip"
    
    # Create directory structure if it doesn't exist
    cord_dir.mkdir(exist_ok=True)
    data_dir = cord_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    for split in ["train", "valid", "test"]:
        split_dir = data_dir / split
        split_dir.mkdir(exist_ok=True)
        (split_dir / "image").mkdir(exist_ok=True)
        (split_dir / "json").mkdir(exist_ok=True)
    
    # Check if data already exists
    if len(list((data_dir / "train" / "image").glob("*.jpg"))) > 0 and len(list((data_dir / "test" / "image").glob("*.jpg"))) > 0:
        logging.info("CORD dataset already properly organized.")
        return
    
    # Extract from archive if requested
    if extract_archives and cord_archive.exists():
        logging.info(f"Extracting CORD dataset from {cord_archive}")
        extract_dir = cord_dir / "temp_extract"
        
        with zipfile.ZipFile(cord_archive, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # CORD structure is usually a folder containing train/valid/test directories
        # Iterate through them and copy files to the right location
        for split in ["train", "valid", "test"]:
            extracted_split_dir = next(extract_dir.glob(f"*/{split}"), None)
            if not extracted_split_dir:
                extracted_split_dir = extract_dir / split
                if not extracted_split_dir.exists():
                    logging.warning(f"Could not find {split} directory in the extracted archive")
                    continue
            
            # Move images
            for img_file in extracted_split_dir.glob("**/image/*.jpg"):
                shutil.copy(img_file, data_dir / split / "image")
            
            # Move annotations
            for json_file in extracted_split_dir.glob("**/json/*.json"):
                shutil.copy(json_file, data_dir / split / "json")
        
        # Clean up
        shutil.rmtree(extract_dir)
    else:
        if not cord_archive.exists() and extract_archives:
            logging.error(f"Archive {cord_archive} not found.")
        if len(list((data_dir / "train" / "image").glob("*.jpg"))) == 0 or len(list((data_dir / "test" / "image").glob("*.jpg"))) == 0:
            logging.error("CORD dataset not properly organized and no archive to extract.")
            return
    
    # Validate and process CORD annotations
    process_cord_annotations(data_dir / "train")
    process_cord_annotations(data_dir / "valid")
    process_cord_annotations(data_dir / "test")
    
    logging.info("CORD dataset processing completed.")


def process_cord_annotations(data_dir):
    """Process CORD annotations to ensure they're in the correct format."""
    json_dir = data_dir / "json"
    image_dir = data_dir / "image"
    
    for annotation_file in json_dir.glob("*.json"):
        with open(annotation_file, "r", encoding="utf-8") as f:
            try:
                annotation = json.load(f)
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing JSON in {annotation_file}: {e}")
                continue
        
        # Check if corresponding image exists
        image_file = image_dir / annotation_file.name.replace(".json", ".jpg")
        if not image_file.exists():
            logging.warning(f"Image file not found: {image_file}")
            continue
        
        # Get image dimensions for bounding box validation
        try:
            img = Image.open(image_file)
            width, height = img.size
        except Exception as e:
            logging.error(f"Error opening image {image_file}: {e}")
            continue
        
        # Check and fix annotation structure if needed
        modified = False
        
        # Ensure "valid_line" field exists
        if "valid_line" not in annotation:
            logging.error(f"Invalid annotation format in {annotation_file}: missing 'valid_line' field")
            continue
        
        # Process each valid line and its words
        for line_idx, line in enumerate(annotation["valid_line"]):
            if "words" not in line:
                logging.warning(f"Line {line_idx} in {annotation_file} has no 'words' field")
                continue
            
            for word_idx, word in enumerate(line["words"]):
                # Ensure 'text' and 'quad' fields exist
                if "text" not in word or "quad" not in word:
                    logging.warning(f"Word {word_idx} in line {line_idx} has missing fields in {annotation_file}")
                    continue
                
                # Validate quad coordinates (array of 4 points with x,y)
                quad = word["quad"]
                if not isinstance(quad, list) or len(quad) != 4:
                    logging.warning(f"Invalid quad format in {annotation_file}, word {word_idx}")
                    continue
                
                # Check and fix each point
                for point_idx, point in enumerate(quad):
                    if "x" not in point or "y" not in point:
                        logging.warning(f"Missing coordinates in quad in {annotation_file}")
                        continue
                    
                    # Ensure x and y are integers
                    if not isinstance(point["x"], int):
                        quad[point_idx]["x"] = int(point["x"])
                        modified = True
                    
                    if not isinstance(point["y"], int):
                        quad[point_idx]["y"] = int(point["y"])
                        modified = True
                    
                    # Ensure coordinates are within image bounds
                    if point["x"] < 0 or point["x"] > width:
                        quad[point_idx]["x"] = max(0, min(width, point["x"]))
                        modified = True
                    
                    if point["y"] < 0 or point["y"] > height:
                        quad[point_idx]["y"] = max(0, min(height, point["y"]))
                        modified = True
        
        # Save modified annotation if changes were made
        if modified:
            with open(annotation_file, "w", encoding="utf-8") as f:
                json.dump(annotation, f, indent=2)
    
    logging.info(f"Processed CORD annotations in {data_dir}")


def process_doclaynet(data_dir, extract_archives=False):
    """Process DocLayNet dataset to ensure it has the correct structure."""
    doclaynet_dir = Path(data_dir) / "doclaynet"
    archives_dir = Path(data_dir) / "archives"
    doclaynet_archive = archives_dir / "doclaynet.zip"
    
    # Create directory structure if it doesn't exist
    doclaynet_dir.mkdir(exist_ok=True)
    test_dir = doclaynet_dir / "test"
    test_dir.mkdir(exist_ok=True)
    (test_dir / "images").mkdir(exist_ok=True)
    (test_dir / "annotations").mkdir(exist_ok=True)
    (test_dir / "images_draw").mkdir(exist_ok=True)
    (test_dir / "pdfs").mkdir(exist_ok=True)
    
    # Check if data already exists
    if len(list((test_dir / "images").glob("*.jpg"))) > 0 or len(list((test_dir / "images").glob("*.png"))) > 0:
        logging.info("DocLayNet dataset already properly organized.")
        return
    
    # Extract from archive if requested
    if extract_archives and doclaynet_archive.exists():
        logging.info(f"Extracting DocLayNet dataset from {doclaynet_archive}")
        extract_dir = doclaynet_dir / "temp_extract"
        
        with zipfile.ZipFile(doclaynet_archive, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Process the extracted data - this is a placeholder since DocLayNet structure varies
        for img_file in extract_dir.glob("**/*.jpg"):
            shutil.copy(img_file, test_dir / "images")
        
        for img_file in extract_dir.glob("**/*.png"):
            shutil.copy(img_file, test_dir / "images")
        
        for pdf_file in extract_dir.glob("**/*.pdf"):
            shutil.copy(pdf_file, test_dir / "pdfs")
        
        for json_file in extract_dir.glob("**/*.json"):
            if "annotations" in str(json_file).lower():
                shutil.copy(json_file, test_dir / "annotations")
        
        # Clean up
        shutil.rmtree(extract_dir)
    else:
        if not doclaynet_archive.exists() and extract_archives:
            logging.error(f"Archive {doclaynet_archive} not found.")
        if len(list((test_dir / "images").glob("*.jpg"))) == 0 and len(list((test_dir / "images").glob("*.png"))) == 0:
            logging.error("DocLayNet dataset not properly organized and no archive to extract.")
            return
    
    # Process annotations
    process_doclaynet_annotations(test_dir)
    
    logging.info("DocLayNet dataset processing completed.")


def process_doclaynet_annotations(data_dir):
    """Process DocLayNet annotations to ensure they're in the correct format."""
    annotations_dir = data_dir / "annotations"
    images_dir = data_dir / "images"
    
    for annotation_file in annotations_dir.glob("*.json"):
        with open(annotation_file, "r", encoding="utf-8") as f:
            try:
                annotation = json.load(f)
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing JSON in {annotation_file}: {e}")
                continue
        
        # Check if corresponding image exists (either jpg or png)
        base_name = annotation_file.stem
        jpg_image = images_dir / f"{base_name}.jpg"
        png_image = images_dir / f"{base_name}.png"
        
        if jpg_image.exists():
            image_file = jpg_image
        elif png_image.exists():
            image_file = png_image
        else:
            logging.warning(f"Image file not found for annotation: {annotation_file}")
            continue
        
        # Get image dimensions for bounding box validation
        try:
            img = Image.open(image_file)
            width, height = img.size
        except Exception as e:
            logging.error(f"Error opening image {image_file}: {e}")
            continue
        
        # Validate DocLayNet annotations structure
        modified = False
        
        # DocLayNet typically has "elements" field containing document components
        if "elements" not in annotation:
            logging.error(f"Invalid annotation format in {annotation_file}: missing 'elements' field")
            continue
        
        for idx, element in enumerate(annotation["elements"]):
            # Check if element has text and bbox
            if "text" not in element or "bbox" not in element:
                continue
            
            # Validate and fix bbox if needed - DocLayNet usually uses [x, y, width, height]
            bbox = element["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                logging.warning(f"Invalid bbox format in {annotation_file}, element {idx}")
                continue
            
            # Convert to integers if not already
            if not all(isinstance(coord, int) for coord in bbox):
                element["bbox"] = [int(coord) for coord in bbox]
                modified = True
            
            # Ensure values are non-negative and within image bounds
            x, y, w, h = element["bbox"]
            if x < 0 or y < 0 or x + w > width or y + h > height or w <= 0 or h <= 0:
                element["bbox"] = [
                    max(0, x),
                    max(0, y),
                    min(width - x if x < width else 1, w),
                    min(height - y if y < height else 1, h)
                ]
                modified = True
        
        # Save modified annotation if changes were made
        if modified:
            with open(annotation_file, "w", encoding="utf-8") as f:
                json.dump(annotation, f, indent=2)
    
    logging.info(f"Processed DocLayNet annotations in {data_dir}")


def main():
    parser = argparse.ArgumentParser(description="Process document understanding datasets.")
    parser.add_argument("--data_dir", type=str, default="data", help="Base data directory")
    parser.add_argument("--dataset", type=str, choices=["funsd", "sroie", "cord", "doclaynet", "all"], 
                        default="all", help="Dataset to process")
    parser.add_argument("--extract", action="store_true", help="Extract from archives if needed")
    args = parser.parse_args()
    
    setup_logging()
    logging.info(f"Processing dataset(s): {args.dataset}")
    
    if args.dataset == "funsd" or args.dataset == "all":
        process_funsd(args.data_dir, args.extract)
    
    if args.dataset == "sroie" or args.dataset == "all":
        process_sroie(args.data_dir, args.extract)
    
    if args.dataset == "cord" or args.dataset == "all":
        process_cord(args.data_dir, args.extract)
    
    if args.dataset == "doclaynet" or args.dataset == "all":
        process_doclaynet(args.data_dir, args.extract)
    
    logging.info("Data processing completed.")


if __name__ == "__main__":
    main()
