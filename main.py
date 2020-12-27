import argparse
import json
import shutil
from pathlib import Path

from PIL import Image
from torch.autograd import Variable
from torchvision import transforms, models


TMP_DIR_PATH = "./tmp"
IMAGENET_LABEL_PATH = "./imagenet_simple_labels.json"
TOP_K = 5


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
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ]
        )
        self.tmp_dir = Path(TMP_DIR_PATH)
        self.tmp_dir.mkdir(exist_ok=True)
        with open(IMAGENET_LABEL_PATH, "r") as f:
            self.imagenet_simple_labels = json.load(f)

    def load(self, image_path: Path, quality: float) -> Image:
        image_filename = image_path.stem
        distorted_filepath = self.tmp_dir / f"{image_filename}_{quality}.jpg"

        image = Image.open(image_path)
        image_rgb = image.convert("RGB")
        image_rgb.save(
            distorted_filepath,
            quality=quality,
            subsampling=0,
        )
        image_distorted = Image.open(distorted_filepath)
        return image_distorted

    def inference(self, image_path: Path, quality: int):
        image = self.load(image_path, quality)
        image = self.transform(image)
        image_batch = image.unsqueeze(0)

        model_input = Variable(image_batch)
        model_output = self.model(model_input)

        model_output_np = model_output[0].detach().numpy()
        label_merged = {
            label: logit for label, logit in zip(
                self.imagenet_simple_labels,
                model_output_np,
            )
        }
        top_k_labels = dict(
            sorted(label_merged.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        )
        print(f"{image_path} / {quality} / {top_k_labels}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=Path,
        required=True,
        help="Local image path to perform inference. png file recommended",
    )
    parser.add_argument(
        "--qualities",
        type=int,
        nargs='+',
        help="List of quality rates to be applied to given image (0 ~ 100)",
    )
    args = parser.parse_args()
    print(args)

    runner = Runner(pretrained=True)
    for quality in args.qualities:
        runner.inference(args.image_path, quality)

