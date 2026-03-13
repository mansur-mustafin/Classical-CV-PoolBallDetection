import argparse
import json
from dataclasses import dataclass
from pathlib import Path

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


def process_image(image_path: Path, config: BallDetectConfig) -> ImageResult:
    # TODO
    return ImageResult(image_path=str(image_path), balls=[])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline pool-ball detector (heuristic)")
    p.add_argument("--input_json", type=Path, help="Path to data/json/input.json")
    p.add_argument("--output-json", type=Path, help="Path to data/json/output.json")
    p.add_argument("--data-dir", type=Path, default=Path("data/"), help="Directory for input images")
    p.add_argument("--topdown-dir", type=Path, default=Path("data/outputs/"), help="Directory for result images")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Load input JSON
    with open(args.input_json, "r") as f:
        input_data: dict = json.load(f)
        image_paths: list[str] = input_data.get("image_path", [])

    results: list[ImageResult] = []
    for img_path in tqdm(image_paths, desc="Processing images"):
        image_result = process_image(Path(img_path), BallDetectConfig())
        results.append(image_result)

    # Save output JSON
    output_data = [result.to_dict() for result in results]
    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    main()
