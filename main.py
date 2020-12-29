import argparse
import itertools
import json
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


TMP_DIR_PATH = "./tmp"
IMAGENET_LABEL_PATH = "./imagenet_simple_labels.json"
TOP_K = 5


def softmax(x: np.array) -> np.array:
    x -= np.max(x)
    return np.exp(x) / sum(np.exp(x))


class DegradedImageDataset(Dataset):
    def __init__(self, image_path_list: List[Path], quality_list: List[int]):
        self.tmp_dir = Path(TMP_DIR_PATH)
        self.tmp_dir.mkdir(exist_ok=True)
        self.items = self.load_items(image_path_list, quality_list)
        self.length = len(image_path_list) * len(quality_list)

    def __len__(self):
        return self.length

    def load_item(self, image_path: Path, quality_percentage: float) -> Image:
        image_filename = image_path.stem
        distorted_filepath = (
            self.tmp_dir / f"{image_filename}_{quality_percentage}.jpg"
        )

        image = Image.open(image_path)
        image_rgb = image.convert("RGB")
        image_rgb.save(
            distorted_filepath, quality=quality_percentage, subsampling=0,
        )
        image_distorted = Image.open(distorted_filepath)
        return image_distorted

    def load_items(
        self, image_path_list: List[Path], quality_list: List[int]
    ) -> List[Tuple[str, int, "Image"]]:
        return [
            (image_path, quality, self.load_item(image_path, quality))
            for image_path, quality in itertools.product(
                image_path_list, quality_list
            )
        ]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.items[idx]


class Runner:
    def __init__(self, pretrained: bool = True):
        self.model = models.mobilenet_v2(pretrained=pretrained)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        with open(IMAGENET_LABEL_PATH, "r") as f:
            self.imagenet_simple_labels = json.load(f)

    def inference(
        self,
        image: Image,
        image_path: Path,
        quality_percentage: int,
        use_softmax: bool,
    ):
        image = self.transform(image)
        image_batch = image.unsqueeze(0)

        model_input = Variable(image_batch)
        model_output = self.model(model_input)

        model_output_np = model_output[0].detach().numpy()
        if use_softmax:
            output = softmax(model_output_np)
        else:
            output = model_output_np

        label_merged = {
            label: value
            for label, value in zip(self.imagenet_simple_labels, output)
        }
        top_k_labels = dict(
            sorted(label_merged.items(), key=lambda x: x[1], reverse=True)[
                :TOP_K
            ]
        )
        print(
            {
                "image_path": str(image_path),
                "quality_percentage": quality_percentage,
                f"top_{TOP_K}_labels": top_k_labels,
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=Path,
        required=True,
        help="Local image path to perform inference. png file recommended",
    )
    parser.add_argument(
        "--quality_percentages",
        type=int,
        nargs="+",
        help="List of quality degradation rate to be applied to given image",
    )
    parser.add_argument(
        "--softmax",
        dest="use_softmax",
        action="store_true",
        help="Whether or not to apply softmax over model output",
    )
    parser.add_argument(
        "--no-softmax", dest="use_softmax", action="store_false",
    )
    parser.set_defaults(use_softmax=False)
    args = parser.parse_args()
    print(args)

    runner = Runner(pretrained=True)
    dataset = DegradedImageDataset([args.image_path], args.quality_percentages)
    for data in dataset:
        image_path, quality_percentage, image = data
        runner.inference(
            image, image_path, quality_percentage, args.use_softmax
        )
