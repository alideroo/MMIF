# MMIF

## Training

### 1 Environment
```
conda create --n mmif python=3.8
conda activate mmif
pip install -r requirements.txt
```

### 2 Data Preparation

Download dataset from [this link](https://github.com/xianming-gu/Havard-Medical-Image-Fusion-Datasets) and place it in the folder 'test_img' and 'train_img'.

### 3 Data Pre-Processing

```
python dataprocessing.py
```
and the processed training dataset is in './data'.

### 4 Training

```
python train.py --gpu_id your_device_id --num_epochs number_of_epochs --epoch_gap number_of_epochs_PhaseI ...
```

## Evaluation

```
python test.py --ckpt_path your_model_path --dataset_name dataset_name
```

## Contact

If you have any questions, please contact rczpku@163.com.
