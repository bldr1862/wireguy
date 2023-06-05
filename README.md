## Install dependencies

```
pip install -r requirements.txt
```

## Extract features

You can modify the backbone size under the line `backbone = BKB.get_backbone(size="s")` in `get_features.py`. Allowed sizes are s, b and l.

```bash
python get_features.py
```


## Train head

Make sure the variables `DATA_PATH` and `FEATURES_PATH` point to the right folder in your environment

```bash
python train_head.py
```

## Colab

To run in colab run the following lines to install the right torch and torchvision version.

```bash 
!pip uninstall -y torch torchvision torchaudio torchtext torchdata
!pip install torch==2.0.0 torchvision
!pip install pandas
!pip install opencv-python==4.7.0.72
!pip install tqdm
!pip install torcheval==0.0.6
!pip install xformers==0.0.18
```