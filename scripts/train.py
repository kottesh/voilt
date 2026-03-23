from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO and export NCNN")
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="voilt-edge")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"

    export_model = YOLO(str(best_weights))
    export_model.export(format="ncnn", imgsz=args.imgsz, half=False)

    print(f"best_pt={best_weights}")
    print(f"ncnn_dir={best_weights.parent / 'best_ncnn_model'}")


if __name__ == "__main__":
    main()
