import itertools
import json
from pathlib import Path
from pprint import pprint
from typing import List, Tuple, Dict, Any

import hydra
import jsonlines
import torch
import numpy as np
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, models


TMP_DIR_PATH = "./tmp"
IMAGENET_LABEL_PATH = "./imagenet_simple_labels.json"
TOP_K = 5


def softmax(x: np.ndarray) -> np.ndarray:
    x -= np.max(x)
    return np.exp(x) / sum(np.exp(x))


class DegradedImageDataset(Dataset):
    def __init__(self, image_path_list: List[str], quality_list: List[int]):
        self.tmp_dir = Path(TMP_DIR_PATH)
        self.tmp_dir.mkdir(exist_ok=True)
        self.items = self.load_items(image_path_list, quality_list)
        self.length = len(image_path_list) * len(quality_list)

    def __len__(self):
        return self.length

    def load_item(self, image_path: str, quality_percentage: float) -> "Image":
        image_filename = Path(image_path).stem
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
        self, image_path_list: List[str], quality_list: List[int]
    ) -> List[Tuple[str, int, "Image"]]:
        return [
            (image_path, quality, self.load_item(image_path, quality))
            for image_path, quality in itertools.product(
                image_path_list, quality_list
            )
        ]

    def __getitem__(self, idx: Any) -> Tuple[str, int, "Image"]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.items[idx]

    def __iter__(self):
        for item in self.items:
            yield item


class Runner:
    def __init__(
        self, label_path: str = IMAGENET_LABEL_PATH, pretrained: bool = True
    ):
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
        with open(label_path, "r") as f:
            self.imagenet_simple_labels = json.load(f)

    def inference(
        self,
        image: "Image",
        image_path: Path,
        quality_percentage: int,
        use_softmax: bool,
    ) -> Dict[str, Any]:
        image = self.transform(image)
        image_batch = image.unsqueeze(0)

        model_input = Variable(image_batch)
        model_output = self.model(model_input)

        model_output_np = model_output[0].detach().numpy()
        output: np.ndarray
        if use_softmax:
            output = softmax(model_output_np)
        else:
            output = model_output_np

        label_merged: Dict[str, float] = {
            label: float(value)
            for label, value in zip(self.imagenet_simple_labels, output)
        }
        top_k_labels = dict(
            sorted(label_merged.items(), key=lambda x: -x[1])[:TOP_K]
        )
        return {
            "image_path": str(image_path),
            "quality_percentage": quality_percentage,
            f"top_{TOP_K}_labels": top_k_labels,
        }


@hydra.main(config_name="config")
def run(_cfg: DictConfig) -> None:
    cwd = hydra.utils.get_original_cwd()

    image_path_list: List[str] = _cfg.config.image_path_list
    quality_percentage_list: List[int] = _cfg.config.quality_percentage_list
    use_softmax: bool = _cfg.config.use_softmax
    output_file_jsonl: str = _cfg.config.output_file_jsonl

    runner = Runner(
        label_path=f"{cwd}/imagenet_simple_labels.json", pretrained=True,
    )
    dataset = DegradedImageDataset(image_path_list, quality_percentage_list)

    with jsonlines.open(f"{cwd}/{output_file_jsonl}", mode="w") as writer:
        for data in dataset:
            image_path, quality_percentage, image = data
            result = runner.inference(
                image, image_path, quality_percentage, use_softmax
            )
            pprint(result)
            writer.write(result)


if __name__ == "__main__":
    run()
