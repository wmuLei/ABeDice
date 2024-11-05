# **1.Environment**

## Please prepare an environment with python=3.9, and then use the command "pip install -r requirements.txt" for the dependencies.

# **2.Pre-training weights we trained**

## Our trained weights are placed in ./results/Dataset/ and are directly available for testing:

```
python test.py --output_dir ./results/DATASET/FILENAME --volume_path ./data/DATASET/ --list_dir ./lists/lists_DATASET --dataset DATASET
```

# **3.Train/Test**

## Run the train script on REFUGE dataset

### Train with ABeDice:

```
python train.py --loss_type ABeDice --d_a 1 --d_b 2 --base_lr 0.1 --output_dir ./results/REFUGE/ABeDice12_lr0.1 --root_path ./data/REFUGE/ --list_dir ./lists/lists_REFUGE --dataset REFUGE
```

### Test:

```
python test.py --output_dir ./results/REFUGE/ABeDice12_lr0.1 --volume_path ./data/REFUGE/ --list_dir ./lists/lists_REFUGE --dataset REFUGE
```

### Train with Dice:

```
python train.py --loss_type Dice --base_lr 0.1 --output_dir ./results/REFUGE/Dice_lr0.1 --root_path ./data/REFUGE/ --list_dir ./lists/lists_REFUGE --dataset REFUGE
```

### Test:

```
python test.py --output_dir ./results/REFUGE/Dice_lr0.1 --volume_path ./data/REFUGE/ --list_dir ./lists/lists_REFUGE --dataset REFUGE
```

## Run the train script on ISIC dataset:

### Train with ABeDice:

```
python train.py --loss_type ABeDice --d_a 2 --d_b 3 --base_lr 0.01 --output_dir ./results/ISIC/ABeDice23_lr0.01 --root_path ./data/ISIC/ --list_dir ./lists/lists_ISIC --dataset ISIC
```

### Test:

```
python test.py --output_dir ./results/ISIC/ABeDice23_lr0.01 --volume_path ./data/ISIC/ --list_dir ./lists/lists_ISIC --dataset ISIC
```

### Train with Dice:

```
python train.py --loss_type Dice --base_lr 0.01 --output_dir ./results/ISIC/Dice_lr0.01 --root_path ./data/ISIC/ --list_dir ./lists/lists_ISIC --dataset ISIC
```

### Test:

```
python test.py --output_dir ./results/ISIC/Dice_lr0.01 --volume_path ./data/ISIC/ --list_dir ./lists/lists_ISIC --dataset ISIC
```

## Run the train script on RITEyes dataset:

### Train with ABeDice:

```
python train.py --loss_type ABeDice --d_a 1 --d_b 1 --base_lr 0.1 --output_dir ./results/RITEyes/ABeDice11_lr0.1 --root_path ./data/RITEyes/ --list_dir ./lists/lists_RITEyes --dataset RITEyes
```

### Test:

```
python test.py --output_dir ./results/RITEyes/ABeDice11_lr0.1 --volume_path ./data/RITEyes/ --list_dir ./lists/lists_RITEyes --dataset RITEyes
```

### Train with Dice:

```
python train.py --loss_type Dice --base_lr 0.1 --output_dir ./results/RITEyes/Dice_lr0.1 --root_path ./data/RITEyes/ --list_dir ./lists/lists_RITEyes --dataset RITEyes
```

### Test:

```
python test.py --output_dir ./results/RITEyes/Dice_lr0.1 --volume_path ./data/RITEyes/ --list_dir ./lists/lists_RITEyes --dataset RITEyes
```
