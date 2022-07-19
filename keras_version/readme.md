# Keras
Scripts to train and test tactip CNN using keras backend. Code has been adapted from Nathan's repo.

## Running code
Install the required dependencies (Nathans repo - [link]) first and make sure the data has been split into a train and test split (see parent folder).
### Training
```
$ python keras/train.py --data_dir $PATH --sensor_type $TACTIP --model_type $model_surface2d --epochs 100
```
Model will be saved in the train folder in the `$PATH`.
### Testing
```
$ python keras/test.py --data_dir $PATH --sensor_type $TACTIP --model_type $model_surface2d
```
MAE will be printed out and also the errors.png will be saved in the test folder in the `$PATH`.
