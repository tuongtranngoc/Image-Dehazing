# Image Dehazing

The implementation of Image Dehazing using UNet-Attention

<p align="center">
    <image src="images/attention_unet.jpg">
</p>


## Environment
The dependencies are listed in `requirements.txt`. Please install and follow the command below:

```bash
pip install -r requirements.txt
```

## Data Preparation
You can download and use NTIRE 2018 datasets including I-HAZE ([Download](http://www.vision.ee.ethz.ch/ntire18/i-haze/)) and O-HAZE ([Download](http://www.vision.ee.ethz.ch/ntire18/o-haze/)) for training and evaluating model. 

After that,
+ Put the data folder under the `dataset` directory
+ Setup config file for each dataset in `src/__init__.py`: replace `CFG_PATH` augment by `src/configs/hehaze.yml`

## Training
Before training, please modify configurations in `src/config/dehaze.yml`
```bash
python -m src.train
```

## Evaluation
```bash
python -m src.evaluate
```


## Prediction
```bash
python -m src.predict
```


## Reference
+ [Unet-Attention paper](https://arxiv.org/abs/1804.03999)
+ 