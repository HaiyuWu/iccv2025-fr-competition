
# DataCV Challenge @ ICCV 2025 📢

This repository provides the FR training code for synthetic FR dataset generation competition in [DataCV workshop @ ICCV2025]().

### What is provided?
- [x] Test sets for competition
- [x] A distributed training framework 
- [x] A list of standard test sets
- [x] A default configuration file for a fair comparison

## 📋Guidance table
<!--ts-->
- [Competition](#competition)
  * [Test resources](#test-resources)
  * [Submission guidance](#submission-guidance)
- [Dataset preparation](#dataset-preparation)
  * [Training sets](#training-sets)
  * [Test sets](#test-sets)
- [Train your own model](#train-your-own-model)
- [Acknowledgement](#acknowledgement)
- [License](#license)
  <!--te-->

## 📦Environment
I suggest you to use Anaconda to better control the environments
```
conda create -n fr_training python=3.8
conda install -n fr_training pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch
conda activate fr_training
```
Then clone the package and use pip to install the dependencies
```
git clone https://github.com/HaiyuWu/iccv2025-fr-competition.git
cd ./iccv2025-fr-competition
pip install -r requirements.txt
```
## ⚔Competition
### Test resources
There are two test sets used in the competition, testA and testB, where **testA** is for getting the sense of the dataset quality and **testB** is for the determination of challenge awards.
- TestA: [Google Drive](https://drive.google.com/file/d/1lnTrlXOOyKA-RcgKxGc-jpugTY6Dsh9y/view?usp=drive_link); [百度云](https://pan.baidu.com/s/1_1Ct3N-igm92e7832iBjsw) [提取码: jyyr]
- TestB: Reserved for now.

### Submission guidance
1. After downloading the testA.xz / testB.xz, please referring to the given command line to extract the test set.
```bash
python extract_test_file.py testA.xz testA
```
2. Train your own face recognition model. Please read [training guidance](#train-your-own-model) carefully!
3. To get the result file for submission, please referring to the given command line to get the result file.
```
python3 get_submission_file.py \
--model_path path/of/the/weights \
--image_paths testA/images.npy
```
4. Once you submit the testA_result.txt or testB_result.txt to [Codalab](), the accuracy will be automatically calculated and reported.
5. Have fun!
## Dataset preparation
### Training sets
We support using .txt file to train the model. Using [file_path_extractor.py](./file_path_extractor.py) to get all the image paths and replacing the path in [./configs/arcface_r50_default.py](./configs/arcface_r50_default.py).
```python
config.train_source = "path/to/.txt/file"
```
**Please ensure that only image files are inside the resource folder when using file_path_extract.py.**
### Test sets
For your convenience, some test sets commonly-used test sets, *e.g.*, LFW, CFP-FP, CALFW, CPLFW, AgeDB-30, can be downloaded [here](https://drive.google.com/file/d/1l7XmqzIZKdKVqu0cOS2EI0bL_9_-wIrc/view?usp=drive_link).
Extract the compressed file then you can simply run [prepare_test_images.py](https://github.com/HaiyuWu/SOTA-FR-train-and-test/blob/main/utils/prepare_test_images.py) to get datasets ready to test
```
python3 utils/prepare_test_images.py \
--xz_folder folder/contains/xz/files \
--destination ./test_set_package_5 \
--datasets lfw cfp_fp agedb_30 calfw cplfw
```
If you use different destination, please change the corresponding configuration in [./configs/arcface_r50_default.py](./configs/arcface_r50_default.py).
## Train your own model
Training command line with 4 GPUs:
```
torchrun --nproc_per_node=4 train.py --config_file ./configs/arcface_r100.py
```
### ❗Note that, you are not supposed to change any training hyperparameters❗

### Acknowledgement
The code is mainly based one [SOTA-Face-Recognition-Train-and-Test](https://github.com/HaiyuWu/SOTA-Face-Recognition-Train-and-Test)


## License
[MIT license](./license.md)

