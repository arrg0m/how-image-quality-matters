# How image quality matters?


## How to (mac)

Make sure you have `anaconda` installed in your Mac environment. [How?](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)


```bash
conda create --name how-image-quality-matters python=3.8 --file requirements.txt
conda activate how-image-quality-matters
python main.py --image_path sample.png --qualities 100 90 80 70
```


## TODO

- Outputs
  - Better aggregation of results
  - Systematic Comparison between outputs
- Use dataloader for efficient collect-and-inference
  - Support for Multiple images
  - Pre-validation of input arguments
  - Batch Inference

- Set of sample images
- Linux & GPU support
- Torchserve Support

- Arguments
  - Logit vs prob

- Seed for non-pretrained model

- Long-term
  - Train with distorted images and then evaluate
  - Fine tune with distorted images and then evaluate
