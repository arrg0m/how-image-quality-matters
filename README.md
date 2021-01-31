# How image quality affects model inference output?


## How to - mac

With appropriate virtual envionment,

```bash
pip install -r requirements.txt
python main.py --image_path sample.png --qualities 100 90 80 70
```


## TODO

- Outputs
  - Better aggregation of results
  - Systematic Comparison between outputs
- Use dataloader for efficient collect-and-inference
  - Pre-validation of input arguments
  - Batch Inference

- Set of "free" sample images
- Linux & GPU support
- Torchserve Support
- load and use custom models & labels

- Seed for non-pretrained model

- CI
  - `flake8`
  - `mypy .`
  - `black . --line-length 80`

- Better way to use hydra
  - [use hydra logging](https://hydra.cc/docs/tutorials/basic/running_your_app/logging)

## References
- https://github.com/anishathalye/imagenet-simple-labels/blob/master/imagenet-simple-labels.json
