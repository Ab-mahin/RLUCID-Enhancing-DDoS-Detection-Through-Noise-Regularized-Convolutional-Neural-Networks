import argparse
import glob
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from util_functions import load_dataset


def evaluate_probs(probs, y, label):
    y_pred = (probs > 0.5).astype(int)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f"{label}: ACC={acc:.4f}, F1={f1:.4f}, TPR={tpr:.4f}, FPR={fpr:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="LUCID ensemble prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List or glob pattern of LUCID .h5 models (e.g. ./output/*-LUCID*.h5)",
    )
    parser.add_argument(
        "--test",
        required=True,
        help="Glob pattern to test HDF5 file (e.g. ./sample-dataset/*-test.hdf5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size for prediction",
    )

    args = parser.parse_args()

    # Resolve model list
    model_paths = []
    for m in args.models:
        if "*" in m or "?" in m:
            model_paths.extend(glob.glob(m))
        else:
            model_paths.append(m)
    model_paths = sorted(list(set(model_paths)))
    if not model_paths:
        raise FileNotFoundError("No models found for patterns: " + " ".join(args.models))

    # Load test set
    X_test, y_test = load_dataset(args.test)
    y_test = y_test.astype(int)

    all_probs = []
    for path in model_paths:
        print(f"Loading model: {path}")
        model = load_model(path)
        probs = model.predict(X_test, batch_size=args.batch_size, verbose=0).reshape(-1)
        all_probs.append(probs)
        evaluate_probs(probs, y_test, label=f"Single model ({path.split('/')[-1]})")

    # Ensemble: average probabilities
    all_probs = np.stack(all_probs, axis=0)  # [num_models, num_samples]
    avg_probs = all_probs.mean(axis=0)
    evaluate_probs(avg_probs, y_test, label=f"Ensemble ({len(model_paths)} models)")


if __name__ == "__main__":
    main()
