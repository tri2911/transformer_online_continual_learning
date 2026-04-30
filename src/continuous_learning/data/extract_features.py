from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from continuous_learning.models.feature_extractor import VGGPlusPlus


def _build_loader(data_root: str, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ]
    )
    dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def extract_all(
    checkpoint_path: str | None,
    output_path: str,
    data_root: str = "./data",
    device: str = "cuda",
    batch_size: int = 512,
    num_workers: int = 4,
) -> None:
    loader = _build_loader(data_root=data_root, batch_size=batch_size, num_workers=num_workers)

    model = VGGPlusPlus().to(device)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint provided; extracting with current VGG++ weights.")
    model.eval()

    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            feats = model(images)
            all_features.append(feats.cpu())
            all_labels.append(labels.cpu())

    features = torch.cat(all_features, dim=0).contiguous()
    labels = torch.cat(all_labels, dim=0).contiguous()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "labels": labels}, out)
    print(f"Saved {tuple(features.shape)} features to {out}")
    print(f"File size: {out.stat().st_size / 1e9:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-extract CIFAR-100 features from VGG++.")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional VGG++ checkpoint path.")
    parser.add_argument("--output", type=str, default="data/features_cache.pt")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    extract_all(
        checkpoint_path=args.checkpoint or None,
        output_path=args.output,
        data_root=args.data_root,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
