import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class BallBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def to_dict(self) -> dict[str, float]:
        return {
            "xmin": np.clip(self.xmin, 0, 1),
            "xmax": np.clip(self.xmax, 0, 1),
            "ymin": np.clip(self.ymin, 0, 1),
            "ymax": np.clip(self.ymax, 0, 1),
        }


@dataclass(frozen=True)
class BallDetection:
    number: int
    box: BallBox

    def to_dict(self) -> dict:
        result = {"number": int(self.number)}
        result.update(self.box.to_dict())
        return result


@dataclass(frozen=True)
class ImageResult:
    image_path: str
    top_down_img: cv2.Mat
    balls: list[BallDetection]

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "num_balls": len(self.balls),
            "balls": [ball.to_dict() for ball in self.balls],
        }


@dataclass(frozen=True)
class BallDetectConfig:
    max_balls: int = 16


def process_image(full_img_path: Path, image_path: Path, config: BallDetectConfig) -> ImageResult:
    img = cv2.imread(str(full_img_path))

    return ImageResult(
        image_path=str(image_path),
        top_down_img=img,
        balls=[]
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline pool-ball detector (heuristic)")
    p.add_argument("--input_json", type=Path, help="Path to data/json/input.json")
    p.add_argument("--output-json", type=Path, help="Path to data/json/output.json")
    p.add_argument("--data-dir", type=Path, default=Path("data/"), help="Directory for input images")
    p.add_argument("--topdown-dir", type=Path, default=Path("data/outputs/"), help="Directory for result images")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    os.makedirs(args.topdown_dir, exist_ok=True)
    os.makedirs(args.output_json.parent, exist_ok=True)

    # Load input JSON
    with open(args.input_json, "r") as f:
        input_data: dict = json.load(f)
        image_paths: list[str] = input_data.get("image_path", [])

    # Process each image and collect results
    results: list[ImageResult] = []
    for img_path in tqdm(image_paths, desc="Processing images"):
        full_img_path = args.data_dir / Path(img_path)
        image_result = process_image(full_img_path, img_path, BallDetectConfig())
        results.append(image_result)

    # Save output JSON
    output_data = [result.to_dict() for result in results]
    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=4)

    # Save result images
    for result in results:
        output_image_path = args.topdown_dir / Path(result.image_path).name
        cv2.imwrite(str(output_image_path), result.top_down_img)


if __name__ == "__main__":
    main()
