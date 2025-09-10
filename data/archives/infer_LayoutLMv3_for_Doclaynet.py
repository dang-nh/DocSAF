import collections
from operator import itemgetter
import torch
import datasets
from transformers import AutoProcessor, AutoModelForTokenClassification
from PIL import Image, ImageDraw
import os
import json
from tqdm import tqdm
from sklearn.metrics import f1_score
import math
import random

    'Formula': 'orange',
    'List-item': 'violet',
    'Page-footer': 'red',
    'Page-header': 'yellow',
    'Picture': 'pink',
    'Section-header': 'purple',
    'Table': 'brown',
    'Text': 'grey',
    'Title': 'cyan'
}


def perturb_bbox(bbox, max_offset=50, iou_threshold=0.7, img_size=(640, 480)):
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1

    img_w, img_h = img_size
    best_bbox = bbox
    for _ in range(10):
        dx, dy = 0, 0
        direction = random.choice(["left", "right", "up", "down"])

        if direction == "left":
            dx = -random.randint(10, max_offset)
        elif direction == "right":
            dx = random.randint(10, max_offset)
        elif direction == "up":
            dy = -random.randint(10, max_offset)
        elif direction == "down":
            dy = random.randint(10, max_offset)

        new_x1 = np.clip(x1 + dx, 0, img_w - width)
        new_y1 = np.clip(y1 + dy, 0, img_h - height)
        new_x2 = new_x1 + width
        new_y2 = new_y1 + height

        new_bbox = [new_x1, new_y1, new_x2, new_y2]
        iou = compute_iou(bbox, new_bbox)

        if iou >= iou_threshold:
            best_bbox = new_bbox
            break

    return best_bbox


# function to sort bounding boxes
def get_sorted_boxes(bboxes):

    # sort by y from page top to bottom 
    sorted_bboxes = sorted(bboxes, key=itemgetter(1), reverse=False)
    y_list = [bbox[1] for bbox in sorted_bboxes]

    # sort by x from page left to right when boxes with same y
    if len(list(set(y_list))) != len(y_list):
        y_list_duplicates_indexes = dict()
        y_list_duplicates = [item for item, count in collections.Counter(y_list).items() if count > 1]
        for item in y_list_duplicates:
            y_list_duplicates_indexes[item] = [i for i, e in enumerate(y_list) if e == item]
            bbox_list_y_duplicates = sorted(np.array(sorted_bboxes, dtype=object)[y_list_duplicates_indexes[item]].tolist(), key=itemgetter(0), reverse=False)
            np_array_bboxes = np.array(sorted_bboxes)
            np_array_bboxes[y_list_duplicates_indexes[item]] = np.array(bbox_list_y_duplicates)
            sorted_bboxes = np_array_bboxes.tolist()

    return sorted_bboxes

def infer(image_path, json_path, output_folder):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with open(json_path, "r", encoding="utf-8") as f:
        data_json = json.load(f)
        form = data_json["form"]
        metadata = data_json["metadata"]

    coco_width, coco_height = metadata["coco_width"], metadata["coco_height"]
    original_width, original_height = metadata["original_width"], metadata["original_height"]

    words = []
    bboxes = []
    labels_char = []
    for item in form:
        words.append(item["text"])
        bboxes.append(convert_box(item["box_line"]))
        labels_char.append(item["category"])

    labels = [label2id[label] for label in labels_char]
    width, height = coco_width, coco_height
    normalized_bboxes = [normalize_bbox(bbox, width, height) for bbox in bboxes]

    perturbed_bboxes = [perturb_bbox(bbox, img_size=(width, height)) for bbox in bboxes]
    normalized_bboxes_perturb = [normalize_bbox(bbox, width, height) for bbox in perturbed_bboxes]

    encoding = processor(
        text=words,
        boxes=normalized_bboxes,
        word_labels=labels,
        images=image,
        return_tensors="pt",
        truncation=True,
        max_length=384,
        padding="max_length"
    )

    encoding_perturbed = processor(
        text=words,
        boxes=normalized_bboxes_perturb,
        word_labels=labels,
        images=image,
        return_tensors="pt",
        truncation=True,
        max_length=384,
        padding="max_length"
    )

    with torch.no_grad():
        outputs = model(**encoding)
        outputs_perturbed = model(**encoding_perturbed)

    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()
    encoding_labels = encoding.labels.squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    logits_perturbed = outputs_perturbed.logits
    predictions_perturbed = logits_perturbed.argmax(-1).squeeze().tolist()
    encoding_labels_perturbed = encoding_perturbed.labels.squeeze().tolist()
    token_boxes_perturbed = encoding_perturbed.bbox.squeeze().tolist()

    true_predictions = [id2label[pred] for pred, label in zip(predictions, encoding_labels) if label != - 100]
    true_labels = [id2label[label] for prediction, label in zip(predictions, encoding_labels) if label != -100]
    true_boxes = [unnormalize_bbox(box, width, height) for box, label in zip(token_boxes, encoding_labels) if
                  label != -100]

    true_boxes_perturbed = [unnormalize_bbox(box, width, height) for box, label in
                            zip(token_boxes_perturbed, encoding_labels_perturbed) if label != -100]
    true_predictions_perturbed = [id2label[pred] for pred, label in
                                  zip(predictions_perturbed, encoding_labels_perturbed) if label != -100]
    true_labels_perturbed = [id2label[label] for label in encoding_labels_perturbed if label != -100]

    # Tính F1-score
    true_labels_idx = [label2id[label] for label in true_labels]
    true_predictions_idx = [label2id[label] for label in true_predictions]

    # Tính F1-score perturbed
    true_labels_idx_perturbed = [label2id[label] for label in true_labels_perturbed]
    true_predictions_idx_perturbed = [label2id[label] for label in true_predictions_perturbed]

    # # Draw
    image_ground_truth = image.copy()
    draw = ImageDraw.Draw(image)
    draw_ground_truth = ImageDraw.Draw(image_ground_truth)

    for predicted_label, box in zip(true_predictions, true_boxes):
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label])

    for label, box in zip(labels_char, bboxes):
        draw_ground_truth.rectangle(box, outline=label2color[label])
        draw_ground_truth.text((box[0] + 10, box[1] - 10), text=label, fill=label2color[label])

    output_path = os.path.join(output_folder, os.path.basename(image_path))
    image_pillow.save(output_path)

    output_ground_truth_path = os.path.join(output_folder, os.path.basename(image_path).split('.')[0] + "_ground_truth.png")
    image_ground_truth.save(output_ground_truth_path)

    return f1_score(true_labels_idx, true_predictions_idx, average="weighted"), f1_score(true_labels_idx_perturbed,
                                                                                         true_predictions_idx_perturbed,
                                                                                         average="weighted")


if __name__ == "__main__":
    datadir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    image_folder = os.path.join(datadir_path, "doclaynet/test/images")
    json_folder = os.path.join(datadir_path, "doclaynet/test/annotations")
    output_folder = os.path.join(datadir_path, "doclaynet/test/images_draw")
    os.makedirs(output_folder, exist_ok=True)

    all_f1 = []
    all_f1_perturbed = []
    for image_name in tqdm(os.listdir(image_folder)):
        json_name = image_name.replace(".png", ".json")

        f1_single, f1_single_perturbed = infer(os.path.join(image_folder, image_name),
                                               os.path.join(json_folder, json_name), output_folder)
        if math.isnan(f1_single) or math.isnan(f1_single_perturbed):
            continue
        all_f1.append(f1_single)
        all_f1_perturbed.append(f1_single_perturbed)

    avg_f1 = sum(all_f1) / len(all_f1)
    avg_f1_perturbed = sum(all_f1_perturbed) / len(all_f1_perturbed)
    print(f"Doclaynet F1 average: {avg_f1:.4f}")
    print(f"Doclaynet F1 perturbed average: {avg_f1_perturbed:.4f}")
    print(f"Doclaynet F1 drop: {(avg_f1 - avg_f1_perturbed):.4f}")
