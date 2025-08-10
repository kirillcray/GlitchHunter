from pathlib import Path
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


def train_patchcore(path_to_data: str = "anomalib/datasets/my_board"):
    # MVTecAD is a popular dataset for anomaly detection
    datamodule = Folder(
        name="my_board",
        root=Path("path_to_data"),
        normal_dir="good",  # Subfolder containing normal images
        abnormal_dir="bad",  # Subfolder containing anomalous images
    )

    # 3. Initialize the model
    # EfficientAd is a good default choice for beginners
    model = Patchcore(
        backbone="wide_resnet50_2",  # Feature extraction backbone
        layers=["layer2", "layer3"],  # Layers to extract features from
        pre_trained=True,  # Use pretrained weights
        num_neighbors=9,  # Number of nearest neighbors
    )

    # Initialize training engine with specific settings
    engine = Engine(
        max_epochs=1,  # Patchcore typically needs only one epoch
        accelerator="auto",  # Automatically detect GPU/CPU
        devices=1,  # Number of devices to use
        default_root_dir="anomalib/results/patchcore/my_board"
    )

    # Train the model
    engine.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    train_patchcore()