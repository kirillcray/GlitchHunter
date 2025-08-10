import argparse
from pathlib import Path

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


def test_anomaly_detector(args):
    engine = Engine(
        max_epochs=1,  # Patchcore typically needs only one epoch
        accelerator="auto",  # Automatically detect GPU/CPU
        devices=1,  # Number of devices to use
        default_root_dir=Path(args.output_path),
    )

    test_data = PredictDataset(
        path=Path(args.input_path),
    )

    model = Patchcore.load_from_checkpoint(args.checkpoint_path)

    predictions = engine.predict(
        model=model,
        dataset=test_data,
    )

    print("\nProcessing Results...")
    if predictions is not None:
        for prediction in predictions:
            image_path = prediction.image_path
            is_anomalous = prediction.pred_label > 0.5

            print(f"Image: {image_path}")
            # print(f"Anomaly Score: {anomaly_score:.3f}")
            print(f"Is Anomalous: {is_anomalous}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run anomaly detection with Patchcore."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the directory containing test images.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="exported_models",
        help="Path to export the model.",
    )
    args = parser.parse_args()

    test_anomaly_detector(args)
