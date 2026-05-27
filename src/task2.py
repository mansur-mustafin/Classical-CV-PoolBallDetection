"""
Requirements:
- numpy==2.4.3
- opencv-python==4.13.0.92
- scipy==1.17.1
- tqdm==4.67.3
- matplotlib==3.10.8
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable, Dict
from typing import List

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm


@dataclass
class ImageResult:
    image_path: str
    balls: int

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "num_balls": self.balls,
        }


@dataclass
class PoolBallCounterConfig:
    # Paths
    image_dir_train: Path
    image_dir_val: Path
    image_dir_test: Path
    output_json: Path

    ckpt_dir: str = "checkpoints"
    seed: int = 42
    num_classes: int = 16
    model_name: str = "resnet18"
    image_size: int = 256

    # Training parameters
    batch_size: int = 16
    num_workers: int = 4
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


def set_seed(seed: int) -> None:
    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)  # same as above


# Models
_MODEL: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    def deco(fn):
        _MODEL[name] = fn
        return fn
    return deco


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in _MODEL:
        return None
    return _MODEL[name](**kwargs)


@register_model("resnet18")
def resnet18(pretrained: bool, num_classes: int = 1) -> nn.Module:
    w = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=w)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


@register_model("resnet50")
def resnet50(pretrained: bool, num_classes: int = 1) -> nn.Module:
    w = models.ResNet50_Weights.DEFAULT if pretrained else None
    m = models.resnet50(weights=w)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


@register_model("efficientnet_v2_s")
def efficientnet_v2_s(pretrained: bool, num_classes: int = 1) -> nn.Module:
    w = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_v2_s(weights=w)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


@register_model("densenet121")
def densenet121(pretrained: bool, num_classes: int = 1) -> nn.Module:
    w = models.DenseNet121_Weights.DEFAULT if pretrained else None
    m = models.densenet121(weights=w)
    in_features = m.classifier.in_features
    m.classifier = nn.Linear(in_features, num_classes)
    return m


@register_model("alexnet")
def alexnet(pretrained: bool, num_classes: int = 1) -> nn.Module:
    w = models.AlexNet_Weights.DEFAULT if pretrained else None
    m = models.alexnet(weights=w)
    in_features = m.classifier[6].in_features
    m.classifier[6] = nn.Linear(in_features, num_classes)
    return m


class PoolBallDataset:
    def __init__(self, image_dir: Path, transform=None, image_size: int = 224) -> None:
        self.image_dir = image_dir
        self.image_paths = sorted(image_dir.glob("*.jpg"))
        self.image_to_balls = {}

        if transform:
            self.transform = transform
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.pytorch.ToTensorV2(),
            ])

        self._load_annotations(image_dir / "_annotations.createml.json")

    def _load_annotations(self, annotation_file: Path) -> None:
        annotations = []

        with open(annotation_file, "r") as f:
            annotations = json.load(f)

        for item in annotations:
            image_path = item["image"]
            num_balls = 0
            for object in item["annotations"]:
                if object["label"] != "Dot":
                    num_balls += 1
            self.image_to_balls[image_path] = num_balls
        
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Any:
        image_path = self.image_paths[idx]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]

        label = self.image_to_balls.get(image_path.name, 0)

        return str(image_path), image, label


class PoolBallCounter:
    def __init__(self, cfg: PoolBallCounterConfig) -> None:
        self.cfg = cfg

        # Datasets and dataloaders
        self.image_dir_train = cfg.image_dir_train
        self.image_dir_val = cfg.image_dir_val
        self.image_dir_test = cfg.image_dir_test

        self.dataset_train = PoolBallDataset(cfg.image_dir_train, image_size=cfg.image_size)
        self.dataset_val = PoolBallDataset(cfg.image_dir_val, image_size=cfg.image_size)
        self.dataset_test = PoolBallDataset(cfg.image_dir_test, image_size=cfg.image_size)

        self.train_dataloader = DataLoader(self.dataset_train, batch_size=cfg.batch_size,
                                           num_workers=cfg.num_workers, shuffle=True)
        self.val_dataloader = DataLoader(self.dataset_val, batch_size=cfg.batch_size,
                                         num_workers=cfg.num_workers, shuffle=False)
        self.test_dataloader = DataLoader(self.dataset_test, batch_size=cfg.batch_size,
                                          num_workers=cfg.num_workers, shuffle=False)

        # Model, optimizer, etc.
        self.model = build_model(cfg.model_name, pretrained=True, num_classes=cfg.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    def train(self) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'mps' if torch.backends.mps.is_available() else
                              'cpu')
        model = self.model.to(device)

        if wandb.run is not None:
            wandb.watch(model, log="gradients", log_freq=100)

        self.ckpt_dir = Path(self.cfg.ckpt_dir) / time.strftime('%Y%m%d-%H%M%S')
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float('inf')

        for epoch in range(self.cfg.num_epochs):
            # Training loop
            model.train()

            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_abs_error = 0.0
            train_sq_error = 0.0

            for image_paths, images, labels in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.cfg.num_epochs}"):
            # for image_paths, images, labels in self.train_dataloader:
                images = images.to(device)
                labels = labels.to(device).long()

                self.optimizer.zero_grad()

                logits = model(images)

                loss = self.criterion(logits, labels)

                loss.backward()
                self.optimizer.step()

                batch_size = images.size(0)
                train_loss += loss.item() * batch_size
                train_total += batch_size

                preds = torch.argmax(logits, dim=1)

                train_correct += (preds == labels).sum().item()
                train_abs_error += torch.abs(preds - labels).sum().item()
                train_sq_error += ((preds - labels) ** 2).sum().item()

            train_loss /= max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)
            train_mae = train_abs_error / max(train_total, 1)
            train_rmse = math.sqrt(train_sq_error / max(train_total, 1))

            # Validation loop
            model.eval()
            
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_abs_error = 0.0
            val_sq_error = 0.0

            with torch.no_grad():
                for image_paths, images, labels in self.val_dataloader:
                    images = images.to(device)
                    labels = labels.to(device).long()

                    logits = model(images)
                    loss = self.criterion(logits, labels)

                    batch_size = images.size(0)
                    val_loss += loss.item() * batch_size
                    val_total += batch_size

                    preds = torch.argmax(logits, dim=1)

                    val_correct += (preds == labels).sum().item()
                    val_abs_error += torch.abs(preds - labels).sum().item()
                    val_sq_error += ((preds - labels) ** 2).sum().item()

            val_loss /= max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)
            val_mae = val_abs_error / max(val_total, 1)
            val_rmse = math.sqrt(val_sq_error / max(val_total, 1))

            if wandb.run is not None:
                wandb.log({
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "train/mae": train_mae,
                    "train/rmse": train_rmse,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/mae": val_mae,
                    "val/rmse": val_rmse,
                    "epoch": epoch + 1,
                }, step=epoch + 1)

            print(
                f"Epoch {epoch + 1}/{self.cfg.num_epochs}\n"
                f"\tTrain Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | MAE: {train_mae:.4f} | RMSE: {train_rmse:.4f}\n"
                f"\tVal Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                save_path = self.ckpt_dir / f"model_epoch_{epoch + 1}_{val_loss:.3f}_{val_mae:.3f}.pth"

                torch.save({
                    "epoch": epoch + 1,
                    "model_name": self.cfg.model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                    "num_classes": self.cfg.num_classes,
                }, save_path)

                print(f"Saved best model to: {save_path}")

    def test(self) -> List[ImageResult]:
        # Placeholder for testing logic
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline pool-ball detector (heuristic)")
    p.add_argument("--image_dir_train", type=Path, help="Path to directory containing input images for training")
    p.add_argument("--image_dir_val", type=Path, help="Path to directory containing input images for validation")
    p.add_argument("--image_dir_test", type=Path, help="Path to directory containing input images for testing")
    p.add_argument("--output_json", type=Path, help="Path to data/json/output.json")
    p.add_argument("--model_name", type=str, default="resnet18", help="Model architecture to use")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    os.makedirs(args.output_json.parent, exist_ok=True)

    config = PoolBallCounterConfig(
        image_dir_train=args.image_dir_train,
        image_dir_val=args.image_dir_val,
        image_dir_test=args.image_dir_test,
        output_json=args.output_json,
        model_name=args.model_name,
    )

    set_seed(config.seed)

    # wandb.login(key=os.getenv("WANDB_API_KEY"))
    # wandb.init(
    #     project="DRL-course",
    #     name=f"run_{time.strftime('%Y%m%d-%H%M%S')}",
    #     config=cfg.to_dict(),
    # )

    counter = PoolBallCounter(cfg=config)
    counter.train()

    results = counter.test()

    print(results)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("Process interrupted by user.")
