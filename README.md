# Classification of Neural Recordings
Neural network to classify spikes in a neural recording from bipolar electrodes.

Follow the instructions below to use the repository.

Written for Python 3.7.5

## 1. Install dependencies
``` shell
pip install -r requirements.txt
```
Depending on your configuration, you may need the `--user` flag.
## 2. Train model (optional)
A pre-trained model is available in `model.h5`
``` shell
python train.py
```
## 3. Predict classes from submission dataset
``` shell
python predict.py
```

## To use a different dataset
1. Save dataset as `data/raw.mat`
2. Split dataset into training, test, and validation by running:
    ``` shell
    python splitData.py
    ```
