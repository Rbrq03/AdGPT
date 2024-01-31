# AdGPT: Explore Meaningful Advertising with ChatGPT

Official repo for AdGPT: Explore Meaningful Advertising with ChatGPT

![](./assert/figure1.png)

## Method

![](./assert/figure2.png)

## Environment Setup

```
conda create -n AdGPT python=3.9
conda activate AdGPT

pip install -r requirements.txt

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Inferrence

```
python demo.py --image_path ./img/test.jpg \
               --openai_key YOUR_API_KEY

#if you have more than one device
CUDA_VISIBLE_DEVICES=0 python demo.py --image_path ./img/test.jpg \
                                      --openai_key YOUR_API_KEY
```

## Acknowledgement

We would like to express our gratitude to previous work, which includes but is not limited to: [EasyOCR](https://github.com/JaidedAI/EasyOCR), [Detic](https://github.com/facebookresearch/Detic), [Transformers](https://github.com/huggingface/transformers), and [detectron2](https://github.com/facebookresearch/detectron2).