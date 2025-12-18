import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from typing import List, Optional, Tuple

import cv2
import joblib
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

CLASS_MAPPING = {
    "cardboard": 0,
    "glass": 1,
    "metal": 2,
    "paper": 3,
    "plastic": 4,
    "trash": 5,
    "unknown": 6,
}


def _collect_image_paths(root_dir: str, recursive: bool = True) -> List[str]:
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"dataFilePath is not a directory: {root_dir}")

    paths: List[str] = []

    if recursive:
        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                if name.lower().endswith(IMAGE_EXTS):
                    paths.append(os.path.join(dirpath, name))
    else:
        for name in os.listdir(root_dir):
            full = os.path.join(root_dir, name)
            if os.path.isfile(full) and name.lower().endswith(IMAGE_EXTS):
                paths.append(full)

    paths.sort()
    return paths


def _get_feature_extractor() -> Model:
    if not hasattr(_get_feature_extractor, "_model"):
        base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base.output)
        _get_feature_extractor._model = Model(base.input, x)  # type: ignore[attr-defined]
    return _get_feature_extractor._model  # type: ignore[attr-defined]


def _to_int_label(pred) -> int:
    if isinstance(pred, (int, np.integer)):
        return int(pred)
    return CLASS_MAPPING.get(str(pred), CLASS_MAPPING["unknown"])



def _apply_rejection(model, X: np.ndarray, threshold: float = 0.6) -> List[int]:

    # Apply confidence-based rejection for SVM or KNN models.

    # Probability-based models (e.g., SVM)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        max_probs = np.max(probs, axis=1)
        preds = model.predict(X)

        return [
            CLASS_MAPPING["unknown"] if conf < threshold else _to_int_label(pred)
            for pred, conf in zip(preds, max_probs)
        ]

    #  KNN vote-based rejection
    if hasattr(model, "kneighbors") and hasattr(model, "n_neighbors"):
        distances, indices = model.kneighbors(X)
        y = getattr(model, "_y", None)
        preds = model.predict(X)

        if y is None:
            return [_to_int_label(p) for p in preds]

        neighbor_labels = y[indices]
        results: List[int] = []

        for i, pred in enumerate(preds):
            votes = np.sum(neighbor_labels[i] == pred)
            confidence = votes / model.n_neighbors
            results.append(
                CLASS_MAPPING["unknown"]
                if confidence < threshold
                else _to_int_label(pred)
            )

        return results

    # Fallback (no rejection)
    return [_to_int_label(p) for p in model.predict(X)]


# Required Public Function
def predict(dataFilePath: str, bestModelPath: str) -> List[int]:


    if not os.path.isfile(bestModelPath):
        raise FileNotFoundError(f"Model file not found: {bestModelPath}")

    # Load trained model
    model = joblib.load(bestModelPath)

    # Load scaler if present
    scaler: Optional[object] = None
    scaler_path = os.path.join(os.path.dirname(bestModelPath), "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    # Collect image paths
    image_paths = _collect_image_paths(dataFilePath, recursive=True)
    if not image_paths:
        return []

    feature_extractor = _get_feature_extractor()

    # Extract CNN features
    features: List[np.ndarray] = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img = preprocess_input(img)

        feat = feature_extractor.predict(img, verbose=0)[0]
        features.append(feat)

    if not features:
        return []

    X = np.asarray(features, dtype=np.float32)
    if scaler is not None:
        X = scaler.transform(X)

    # Predict with rejection
    return _apply_rejection(model, X, threshold=0.6)



if __name__ == "__main__":
    dataFilePath = r"..\test"
    bestModelPath = r"..\models\svm_model.pkl"
    print("\nLoading and Predicting: ")
    preds = predict(dataFilePath, bestModelPath)

    id_to_class = {v: k for k, v in CLASS_MAPPING.items()}
    image_paths = _collect_image_paths(dataFilePath, recursive=True)
    print("\nPrediction Results:")
    print("-" * 70)

    for i, (img_path, pred_id) in enumerate(zip(image_paths, preds), start=1):
        label = id_to_class.get(pred_id, "unknown")
        img_name = os.path.basename(img_path)
        print(f"{i:03d}. {img_name:<25} → {label} ({pred_id})")

    print("-" * 70)
    print(f"Total images processed: {len(preds)}")