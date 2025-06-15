import json
from collections import Counter

def compute_class_weights(json_path, num_classes=5, smoothing=None):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Count number of objects per class
    class_counts = Counter()
    for ann in data['annotations']:
        class_id = ann['category_id']
        class_counts[class_id] += 1

    print("Class Counts:", dict(class_counts))

    # Fill missing classes with 0 count
    for i in range(num_classes):
        if i not in class_counts:
            class_counts[i] = 0

    # Avoid division by zero
    min_count = min([count for count in class_counts.values() if count > 0])
    safe_counts = {cls: count if count > 0 else min_count for cls, count in class_counts.items()}

    # Compute inverse frequency weights
    max_count = max(safe_counts.values())
    raw_weights = {cls: max_count / count for cls, count in safe_counts.items()}

    # Optional smoothing
    if smoothing == "log":
        weights = {cls: round(math.log(1 + w), 4) for cls, w in raw_weights.items()}
    elif smoothing == "sqrt":
        weights = {cls: round(w ** 0.5, 4) for cls, w in raw_weights.items()}
    else:
        weights = {cls: round(w, 4) for cls, w in raw_weights.items()}

    # Convert to list (index = class ID)
    weight_list = [weights[i] for i in range(num_classes)]
    return weight_list

# Example usage
if __name__ == "__main__":
    import math
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to train.json (COCO format)")
    parser.add_argument("--smoothing", choices=["log", "sqrt"], help="Apply smoothing to weights")
    args = parser.parse_args()

    weights = compute_class_weights(args.json_path, smoothing=args.smoothing)
    print("Class Weights:", weights)
