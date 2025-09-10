import glob
import json
import os
from pathlib import Path
import pickle
import string
import re
import numpy as np
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm
from PIL import Image

label2color = {
    'O': 'blue',
    'B-COMPANY': 'green', 'I-COMPANY': 'green',
    'B-DATE': 'orange', 'I-DATE': 'orange',
    'B-ADDRESS': 'violet', 'I-ADDRESS': 'violet',
    'B-TOTAL': 'red', 'I-TOTAL': 'red'
}

def subfinder(words_list, answer_list):
    """
    Find answer_list (a sequence of words) in words_list
    Returns: match (True/False), start_idx, end_idx
    """
    # If either list is empty, return no match
    if not words_list or not answer_list:
        return False, 0, 0
        
    for i in range(len(words_list)):
        if words_list[i] == answer_list[0] and words_list[i:i+len(answer_list)] == answer_list:
            return True, i, i+len(answer_list)-1
    return False, 0, 0

def normalize_bbox(bbox, width, height):
    bbox = [int(1000 * (bbox[0] / width)),
            int(1000 * (bbox[1] / height)),
            int(1000 * (bbox[2] / width)),
            int(1000 * (bbox[3] / height))]
    
    # Clip bbox values to ensure they stay within valid embedding range
    # LayoutLMv2/v3 position embeddings typically have max size of 1024 (indices 0-1023)
    # We need to ensure height (bbox[3] - bbox[1]) and width (bbox[2] - bbox[0]) 
    # calculations don't exceed this range
    max_position = 1023  # Maximum valid position embedding index
    
    # Clip each coordinate to valid range
    bbox = [min(max(coord, 0), max_position) for coord in bbox]
    
    # Additional safety check to ensure height and width are within bounds
    # If height or width would exceed max_position, scale down proportionally
    height_val = bbox[3] - bbox[1] 
    width_val = bbox[2] - bbox[0]
    
    if height_val > max_position:
        # Scale down y coordinates proportionally
        scale_factor = max_position / height_val
        bbox[1] = int(bbox[1] * scale_factor)
        bbox[3] = int(bbox[3] * scale_factor)
    
    if width_val > max_position:
        # Scale down x coordinates proportionally  
        scale_factor = max_position / width_val
        bbox[0] = int(bbox[0] * scale_factor)
        bbox[2] = int(bbox[2] * scale_factor)
    
    # Final clipping to ensure all values are within range
    bbox = [min(max(coord, 0), max_position) for coord in bbox]
    
    # Ensure bbox coordinates are valid (x1 < x2, y1 < y2)
    if bbox[0] >= bbox[2]:
        bbox[2] = min(bbox[0] + 1, max_position)
    if bbox[1] >= bbox[3]:
        bbox[3] = min(bbox[1] + 1, max_position)
    
    assert 0 <= bbox[0] <= max_position and 0 <= bbox[1] <= max_position and 0 <= bbox[2] <= max_position and 0 <= bbox[3] <= max_position
    assert bbox[0] < bbox[2] and bbox[1] < bbox[3]
    return bbox


def compute_iou(bb1, bb2, iou_type="both"):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

    Args:
        bb1 : list ['x1', 'y1', 'x2', 'y2']
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : list ['x1', 'y1', 'x2', 'y2']
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        iou_type: str: Cách tính iou
            "both": tính IOU trên tổng 2 bb
            "first": tính IOU trên bb số 1
            "second": tính IOU trên bb số 2
            "compare": lấy IOU lớn nhất trên 2 bbox
        
    Returns:
        float: IoU value in range [0, 1]
    """
    if not all([bb1[0] < bb1[2], bb1[1] < bb1[3], bb2[0] < bb2[2], bb2[1] < bb2[3]]):
        return 0.0

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    if iou_type == "both":
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    elif iou_type == "first":
        iou = intersection_area / float(bb1_area)
    elif iou_type == "compare":
        iou = max(
            [
                intersection_area / float(bb1_area),
                intersection_area / float(bb2_area),
            ]
        )
    else:  # iou_type = "second"
        iou = intersection_area / float(bb2_area)

    return max(0.0, min(iou, 1.0))


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)


def get_line_bbox(bboxs):
    x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
    y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

    assert x1 >= x0 and y1 >= y0
    bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
    return bbox


def normalize_answer(answer):
    """Normalize answer for fair comparison in ANLS metric."""
    # Convert to lowercase
    answer = answer.lower()
    
    # Remove punctuation and special chars
    answer = re.sub(f'[{re.escape(string.punctuation)}]', '', answer)
    
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    
    return answer


def compute_anls(pred, gt, threshold=0.5):
    """
    Compute Answer Normalized Levenshtein Similarity (ANLS) between prediction and ground truth.
    
    Args:
        pred (str): Predicted answer
        gt (str): Ground truth answer
        threshold (float): Threshold below which the answer is considered incorrect
        
    Returns:
        float: ANLS score (if below threshold, returns 0)
    """
    # Normalize answers
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    
    # If either one is empty after normalization, apply special logic
    if len(pred_norm) == 0 and len(gt_norm) == 0:
        return 1.0
    if len(pred_norm) == 0 or len(gt_norm) == 0:
        return 0.0
    
    # Calculate normalized Levenshtein similarity
    max_len = max(len(pred_norm), len(gt_norm))
    lev_dist = levenshtein_distance(pred_norm, gt_norm)
    sim = 1.0 - (lev_dist / max_len)
    
    return sim


def compute_batch_anls(preds, gts, threshold=0.5):
    """
    Compute ANLS for a batch of predictions and ground truths.
    
    Args:
        preds (list): List of predicted answers
        gts (list): List of ground truth answers
        threshold (float): Threshold for ANLS
        
    Returns:
        float: Average ANLS over the batch
    """
    scores = []
    for pred, gt in zip(preds, gts):
        score = compute_anls(pred, gt, threshold)
        scores.append(score)
    
    return np.mean(scores)

def read_pgd_examples(ann_dir, dataset_name, is_normalize=True):
    """Read PGD dataset into a list of examples."""
    examples = []
    label_map = []
    
    data_dir = os.path.dirname(os.path.dirname(ann_dir))
    for filename in tqdm(os.listdir(ann_dir)):
        with open(os.path.join(ann_dir, filename), "r", encoding="utf8") as f:
            data = json.load(f)
        image_paths = glob.glob(os.path.join(data_dir, "images" if dataset_name in ["funsd", "sroie"] else "image", filename.replace(".json", ".*")))
        image_path = [path for path in image_paths if ".json" not in path][0]
        
        image, size = load_image(image_path)
        bboxes = [normalize_bbox(item["adv_box"], size[0], size[1]) if is_normalize else item["adv_box"] for item in data]
        words = [item["word"] for item in data]
        labels = [item["label"] for item in data]
        examples.append({"image": image, "bboxes": bboxes, "words": words, "ner_tags": labels, "image_path": image_path})
        label_map.extend(labels)
    return examples, list(set(label_map))

def read_funsd_examples(data_dir, split, line_level=True, is_normalize=True, use_pgd_boxes=False, iou_threshold=0.9, model_type="v3"):
    """Read FUNSD dataset into a list of examples."""
    ann_dir = os.path.join(data_dir, split, "annotations")
    img_dir = os.path.join(data_dir, split, "images")
    examples = []
    label_map = []
    num_bboxes = 0
    prefix = "" if model_type == "v3" else "layoutlmv2_"
    pgd_ann_dir = os.path.join(data_dir, split, f"{prefix}pgd_adv_annotations_iou_{iou_threshold}", "line_level" if line_level else "word_level")
    if use_pgd_boxes:
        return read_pgd_examples(pgd_ann_dir, "funsd", is_normalize)
    for guid, file in enumerate(sorted(os.listdir(ann_dir))):
        tokens = []
        bboxes = []
        ner_tags = []

        file_path = os.path.join(ann_dir, file)
        with open(file_path, "r", encoding="utf8") as f:
            data = json.load(f)
        image_path = os.path.join(img_dir, file)
        image_path = image_path.replace("json", "png")
        image, size = load_image(image_path)
        for item in data["form"]:
            level_bboxes = []
            words, label = item["words"], item["label"]
            words = [w for w in words if w["text"].strip() != ""]
            if len(words) == 0:
                continue
            if label == "other":
                for w in words:
                    tokens.append(w["text"])
                    ner_tags.append("O")
                    level_bboxes.append(w["box"])
            else:
                tokens.append(words[0]["text"])
                ner_tags.append("B-" + label.upper())
                level_bboxes.append(words[0]["box"])
                for w in words[1:]:
                    tokens.append(w["text"])
                    ner_tags.append("I-" + label.upper())
                    level_bboxes.append(w["box"])
            if line_level:
                level_bboxes = get_line_bbox(level_bboxes)
            if is_normalize:
                level_bboxes = [normalize_bbox(bbox, size[0], size[1]) for bbox in level_bboxes]
            unique_bboxes = []
            for bbox in level_bboxes:
                if bbox not in unique_bboxes:
                    unique_bboxes.append(bbox)
            num_bboxes += len(unique_bboxes)
            bboxes.extend(level_bboxes)
        examples.append(
            {"words": tokens, "bboxes": bboxes, "ner_tags": ner_tags, "image": image, "image_path": image_path})
        label_map.extend(ner_tags)
    print(f"Number of {split} bboxes: {num_bboxes}")
    return examples, list(set(label_map))


def quad_to_box(quad):
    # test 87 is wrongly annotated
    box = (
        max(0, quad["x1"]),
        max(0, quad["y1"]),
        quad["x3"],
        quad["y3"]
    )
    if box[3] < box[1]:
        bbox = list(box)
        tmp = bbox[3]
        bbox[3] = bbox[1]
        bbox[1] = tmp
        box = tuple(bbox)
    if box[2] < box[0]:
        bbox = list(box)
        tmp = bbox[2]
        bbox[2] = bbox[0]
        bbox[0] = tmp
        box = tuple(bbox)
    return box


def read_cord_examples(data_dir, split, line_level=True, is_normalize=True, use_pgd_boxes=False, iou_threshold=0.9, model_type="v3"):
    """Read CORD dataset into a list of examples."""
    examples = []
    labels = []
    ann_dir = os.path.join(data_dir, split, "json")
    prefix = "" if model_type == "v3" else "layoutlmv2_"
    pgd_ann_dir = os.path.join(data_dir, split, f"{prefix}pgd_adv_annotations_iou_{iou_threshold}", "line_level" if line_level else "word_level")
    if use_pgd_boxes:
        return read_pgd_examples(pgd_ann_dir, "cord", is_normalize)
    
    
    img_dir = os.path.join(data_dir, split, "image")
    num_bboxes = 0
    for guid, file in enumerate(sorted(os.listdir(ann_dir))):
        words = []
        bboxes = []
        ner_tags = []
        file_path = os.path.join(ann_dir, file)
        with open(file_path, "r", encoding="utf8") as f:
            data = json.load(f)
        image_path = os.path.join(img_dir, file)
        image_path = image_path.replace("json", "png")
        image, size = load_image(image_path)
        for item in data["valid_line"]:
            level_bboxes = []
            line_words, label = item["words"], item["category"]
            line_words = [w for w in line_words if w["text"].strip() != ""]
            if len(line_words) == 0:
                continue
            if label == "other":
                for w in line_words:
                    words.append(w["text"])
                    ner_tags.append("O")
                    level_bboxes.append(quad_to_box(w["quad"]))
            else:
                words.append(line_words[0]["text"])
                ner_tags.append("B-" + label.upper())
                level_bboxes.append(quad_to_box(line_words[0]["quad"]))
                for w in line_words[1:]:
                    words.append(w["text"])
                    ner_tags.append("I-" + label.upper())
                    level_bboxes.append(quad_to_box(w["quad"]))
            # by default: --segment_level_layout 1
            # if do not want to use segment_level_layout, comment the following line
            if line_level:
                level_bboxes = get_line_bbox(level_bboxes)
            if is_normalize:
                level_bboxes = [normalize_bbox(bbox, size[0], size[1]) for bbox in level_bboxes]
            bboxes.extend(level_bboxes)
            unique_bboxes = []
            for bbox in level_bboxes:
                if bbox not in unique_bboxes:
                    unique_bboxes.append(bbox)
            num_bboxes += len(unique_bboxes)
        labels.extend(ner_tags)
        examples.append({
            "image_path": image_path,
            "words": words,
            "bboxes": bboxes,
            "ner_tags": ner_tags,
            "image": image
        })
    print(f"Number of {split} bboxes: {num_bboxes}")
    return examples, list(set(labels))


def read_sroie_examples(data_dir, split, line_level=True, is_normalize=True, use_pgd_boxes=False, iou_threshold=0.9, model_type="v3"):
    """Read SROIE dataset into a list of examples."""
    examples = []
    total_labels = []

    base_dir = Path(data_dir) / split
    image_dir = base_dir / "images"
    annotation_dir = base_dir / "tagged"
    ocr_dir = base_dir / "ocr"
    prefix = "" if model_type == "v3" else "layoutlmv2_"
    pgd_ann_dir = os.path.join(data_dir, split, f"{prefix}pgd_adv_annotations_iou_{iou_threshold}", "line_level" if line_level else "word_level")
    if use_pgd_boxes:
        return read_pgd_examples(pgd_ann_dir, "sroie", is_normalize)
    for filename in os.listdir(annotation_dir):
        # try:
        if filename.endswith(".json"):
            image_path = str(image_dir / filename.replace(".json", ".jpg"))
            image, size = load_image(image_path)
            annotation_path = str(annotation_dir / filename)
            ocr_path = str(ocr_dir / filename.replace(".json", ".txt"))
            with open(annotation_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            words = data["words"]
            bboxes = data["bbox"]
            labels = data["labels"]

            with open(ocr_path, "r", encoding="utf-8") as f:
                ocr_data = f.readlines()

            if line_level:
                line_bboxes = []

                for word, bbox in zip(words, bboxes):
                    has_line_bbox = False
                    for line in ocr_data:
                        ocr_bbox = [int(line.split(",")[:8][0]), int(line.split(",")[:8][1]),
                                    int(line.split(",")[:8][4]), int(line.split(",")[:8][5])]
                        if compute_iou(bbox, ocr_bbox, iou_type="first") > 0.7:
                            line_bboxes.append(ocr_bbox)
                            has_line_bbox = True
                            break
                    if not has_line_bbox:
                        line_bboxes.append(bbox)
                bboxes = line_bboxes

            # Draw image
            # image_draw_dir = base_dir / "images_draw" / "line" if line_level else base_dir / "images_draw" / "word"
            # os.makedirs(image_draw_dir, exist_ok=True)
            # image_draw_path = image_draw_dir / filename.replace(".json", ".png")
            # draw = ImageDraw.Draw(image)
            # for label, bbox in zip(labels, bboxes):
            #     draw.rectangle(bbox, outline=label2color[label])
            #     draw.text((bbox[0], bbox[1]), label, fill=label2color[label])
            # image.save(image_draw_path)

            if is_normalize:
                normalized_bboxes = [normalize_bbox(bbox, size[0], size[1]) for bbox in bboxes]
            else:
                normalized_bboxes = bboxes

            examples.append({
                "image": image,
                "words": words,
                "bboxes": normalized_bboxes,
                "ner_tags": labels,
                "image_path": image_path
            })

            total_labels.extend(labels)
            # assert len(words) == len(bboxes) == len(labels)
    # except Exception as e:
    #     print(f"Error reading ocr file: {ocr_path}")
    #     print()
    #     print(e)
    return examples, list(set(total_labels))


def read_doclaynet_examples(data_dir, split):
    """Read DocLayNet dataset into a list of examples."""
    examples = []

    base_dir = Path(data_dir) / "doclaynet" / "test"  # DocLayNet typically only has test in publicly available version
    image_dir = base_dir / "images"
    annotation_dir = base_dir / "annotations"

    for filename in os.listdir(annotation_dir):
        if filename.endswith(".json"):
            image_path = str(image_dir / filename.replace(".json", ".jpg"))
            if not os.path.exists(image_path):
                image_path = str(image_dir / filename.replace(".json", ".png"))

            annotation_path = str(annotation_dir / filename)

            with open(annotation_path, "r", encoding="utf-8") as f:
                annotation = json.load(f)

            words = []
            bboxes = []
            labels = []

            for item in annotation["elements"]:
                if "text" in item and "bbox" in item:
                    text = item["text"]
                    bbox = item["bbox"]  # Typically [x1, y1, width, height]

                    # Convert to [x1, y1, x2, y2] format
                    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

                    category = item.get("category", "other")
                    # Map category to index (customize based on your specific set)
                    label_map = {
                        "text": 0,
                        "title": 1,
                        "list": 2,
                        "table": 3,
                        "figure": 4,
                        "other": 5
                    }
                    label_idx = label_map.get(category, 5)

                    words.append(text)
                    bboxes.append(bbox)
                    labels.append(label_idx)

            if words:  # Only add if we found text elements
                examples.append({
                    "image_path": image_path,
                    "words": words,
                    "bboxes": bboxes,
                    "labels": labels
                })

    return examples, None

def is_valid_example(answers, question, words):
    for answer in answers:
        answer_norm = answer.lower().strip()
        text_norm = " ".join(words[:512]).lower().strip()
        if answer_norm in text_norm:
            return True
    return False

def read_docvqa_examples(data_dir, split, line_level=True, is_normalize=True):
    """
    Read DocVQA dataset into a list of examples with questions and answers.
    
    Args:
        data_dir: Path to DocVQA directory
        split: Split name (train, val, test)
        is_normalize: Whether to normalize bounding box coordinates
        
    Returns:
        List of examples with questions and answers
    """
    examples = []
    
    # Define paths
    ann_file = os.path.join(data_dir, f"{split}_v1.0_withQT.json")
    img_dir = os.path.join(data_dir, "images")
    
    # Load annotations
    with open(ann_file, "r", encoding="utf8") as f:
        annotations = json.load(f)
    
    examples = []
    num_bboxes = 0
    # Process annotations
    anno_group_by_image = {}
    for item in annotations["data"]:
        image_filename = item["image"]
        # Full path to image
        if image_filename not in anno_group_by_image:
            anno_group_by_image[image_filename] = []
        anno_group_by_image[image_filename].append(item)
    
    
    for image_filename, items in tqdm(anno_group_by_image.items(), desc="Processing annotations"):
        image_path = os.path.join(img_dir, os.path.basename(image_filename))
        image, size = load_image(image_path)
        
        for item in items:
            question_id = item["questionId"]
            question = item["question"]
            doc_id = item["docId"]
            answers = item["answers"]  # List of correct answers
            
            width = size[0]
            height = size[1]
        
            ocr_file = os.path.join(data_dir, "ocrs", f"{os.path.basename(image_filename).split('.')[0]}.json")
            
            # try:
            with open(ocr_file, "r", encoding="utf8") as f:
                ocr_data = json.load(f)
        
            words = []
            word_bboxes = []
            line_bboxes = []
            
            lines = ocr_data.get("recognitionResults", [{}])[0].get("lines", [])
            for line in lines:
                line_text = line.get("text", "")
                line_bbox = line.get("boundingBox", [0, 0, 0, 0])
                line_bbox = [line_bbox[0], line_bbox[1], line_bbox[4], line_bbox[5]]
                
                if is_normalize:
                    line_bbox = normalize_bbox(line_bbox, width, height)
                
                if "words" in line:
                    for word in line["words"]:
                        word_text = word.get("text", "")
                        word_bbox = word.get("boundingBox", [0, 0, 0, 0])
                        word_bbox = [word_bbox[0], word_bbox[1], word_bbox[4], word_bbox[5]]
                        if is_normalize:
                            word_bbox = normalize_bbox(word_bbox, width, height)
                        
                        words.append(word_text)
                        word_bboxes.append(word_bbox)
                        line_bboxes.append(line_bbox)
                else:
                    line_words = line_text.split()
                    for word in line_words:
                        words.append(word)
                        word_bboxes.append(line_bbox)
                        line_bboxes.append(line_bbox)
            
            is_valid = is_valid_example(answers, question, words)
            if not is_valid:
                continue
            
            bboxes = line_bboxes if line_level else word_bboxes
            
            unique_bboxes = []
            for bbox in bboxes:
                if bbox not in unique_bboxes:
                    unique_bboxes.append(bbox)
            num_bboxes += len(unique_bboxes)
            # Create example
            example = {
                # "image": image,
                "image_path": image_path,
                "words": words,
                "bboxes": bboxes,
                "question": question,
                "answers": answers,
                "question_id": question_id,
                "doc_id": doc_id
            }
            
            examples.append(example)
        
    print(f"Number of {split} bboxes: {num_bboxes}")
    return examples


def extract_entity_text(words, ner_tags, entity_type):
    """
    Extract text corresponding to a specific entity type from document words.
    
    Args:
        words: List of words in the document
        ner_tags: List of NER tags for each word
        entity_type: Entity type to extract (e.g., "TOTAL", "DATE")
        
    Returns:
        Extracted text as a string
    """
    entity_texts = []
    current_entity = []
    
    for word, tag in zip(words, ner_tags):
        if tag == f"B-{entity_type}":
            if current_entity:
                entity_texts.append(" ".join(current_entity))
                current_entity = []
            current_entity.append(word)
        elif tag == f"I-{entity_type}" and current_entity:
            current_entity.append(word)
        elif tag != f"I-{entity_type}" and current_entity:
            entity_texts.append(" ".join(current_entity))
            current_entity = []
    
    if current_entity:
        entity_texts.append(" ".join(current_entity))
    
    return entity_texts[0] if entity_texts else "Not found"

read_examples_func = {
    "funsd": read_funsd_examples,
    "cord": read_cord_examples,
    "sroie": read_sroie_examples,
    "doclaynet": read_doclaynet_examples,
    "docvqa": read_docvqa_examples,
}

def make_docvqa_jsonl_file(data_dir, split, line_level=True, is_normalize=True):
    examples = read_docvqa_examples(data_dir, split, line_level, is_normalize)
    level = "line_level" if line_level else "word_level"
    with open(os.path.join(data_dir, f"{split}_docvqa_{level}.jsonl"), "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    
def read_docvqa_jsonl_file(data_dir, split, line_level=True):
    level = "line_level" if line_level else "word_level"
    with open(os.path.join(data_dir, f"{split}_docvqa_{level}.jsonl"), "r") as f:
        examples = [json.loads(line) for line in f]
        # if split == "val":
        #     examples = examples[:400]
        # elif split == "train":
        #     examples = examples[:3200]
    return examples
    

if __name__ == "__main__":
    # val_examples = read_docvqa_examples("data/docvqa", "val", line_level=False)
    # print(len(val_examples))
    # train_examples = read_docvqa_examples("data/docvqa", "train", line_level=False)
    # print(len(train_examples))
    
    
    # make_docvqa_jsonl_file("data/docvqa", "val", line_level=False)
    # make_docvqa_jsonl_file("data/docvqa", "train", line_level=False)

    make_docvqa_jsonl_file("data/docvqa", "val", line_level=True)
    make_docvqa_jsonl_file("data/docvqa", "train", line_level=True)