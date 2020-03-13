# RRDNet
Implementation of "Zero-Shot Restoration of Underexposed Images via Robust Retinex Decomposition, International Conference on Multimedia and Expo, 2020"

### 1. Prerequisites

* Python 3
* PyTorch >= 0.4.1
* PIL >= 6.1.0
* Opencv-python>=3.4

### 2. Run the project

There are two test underexposed images in `test` folder for demo.

Run

```Cmd
python3 pipline.py
```

and you will get result images in `test` folder.

You can change the path of input image in `conf.py`:

```python
test_image_path = '/the/path/to/input/image'
```





