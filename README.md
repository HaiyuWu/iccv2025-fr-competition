# DataCV Challenge @ ICCV 2025 📢

This repository provides the FR training code for synthetic FR dataset generation competition in [DataCV workshop @ ICCV2025](https://sites.google.com/view/datacv-iccv25/challenge).

### What is provided?
- [x] Test sets for competition ([click](#competition))
- [x] A distributed training framework
- [x] A list of synthetic FR training sets ([click](#dataset-preparation))
- [x] A list of standard test sets ([click](#test-sets))
- [x] A default configuration file for a fair comparison

## 📋Guidance table
<!--ts-->
- [Competition](#competition)
  * [Test resources](#test-resources)
  * [Submission guidance](#submission-guidance)
- [Dataset preparation](#dataset-preparation)
  * [Training sets](#training-sets)
  * [Test sets](#test-sets)
- [FR model training](#fr-model-training)
- [Acknowledgement](#acknowledgement)
- [Q&A](#qa)
- [License](#license)
  <!--te-->

## 📦Environment
I suggest you use Anaconda to better control the environments
```
conda create -n fr_training python=3.10 -y
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
There are two datasets used in the competition, validation and test, where **validation** is for getting the sense of the dataset quality and **test** is for the determination of challenge awards.
- Validation: [Google Drive](https://drive.google.com/file/d/1lnTrlXOOyKA-RcgKxGc-jpugTY6Dsh9y/view?usp=drive_link); [百度云](https://pan.baidu.com/s/1_1Ct3N-igm92e7832iBjsw) [提取码: jyyr]
- Test: Reserved for now.
### Synthetic dataset generation requirements
- The dataset scale can be 10K IDs (up to 0.5M images), 20K IDs (up to 1M images), and 100K IDs (up to 5M images).
- **No real identities.**
### Submission guidance
1. After downloading the validation.xz / test.xz, please referring to the given command line to extract the test set.
```bash
python extract_test_file.py validation.xz validation
```
2. Train your own face recognition model. Please read [training guidance](#fr-model-training) carefully!
3. To get the result file for submission, please referring to the given command line to get the result file.
```
python3 get_submission_file.py \
--model_path path/of/the/weights \
--image_paths validation/images.npy \
--dataset_scale 10K
```
An example of the content in validation_[scale]_result.txt
```text
1.663888216018676758e+00
1.565627694129943848e+00
1.020695686340332031e+00
8.518574833869934082e-01
1.675183534622192383e+00
1.410897254943847656e+00
1.020229220390319824e+00
1.823374867439270020e+00
9.912635087966918945e-01
1.255276918411254883e+00
7.515375614166259766e-01
7.309133410453796387e-01
8.637146949768066406e-01
1.688967466354370117e+00
6.840156912803649902e-01
6.732909083366394043e-01
1.803350210189819336e+00
1.492582917213439941e+00
1.960618495941162109e+00
1.124273777008056641e+00
6.896349191665649414e-01
9.023406505584716797e-01
6.719210743904113770e-01
1.148102045059204102e+00
1.008487701416015625e+00
1.698274970054626465e+00
8.933541774749755859e-01
8.956598639488220215e-01
```
4. Compress the validation_[scale]_result.txt to a .zip file, e.g., result.zip.
5. Submit the .zip file to [Codalab](https://codalab.lisn.upsaclay.fr/competitions/22954) and the accuracy will be automatically calculated and reported. **Note that datasets with different scales will be reported separately.**
6. Have fun!
## Dataset preparation
For your convenience, we provide the link of existing synthetic FR datasets for you to have an easy start. The accuracy values of 0.5M scale are copied from the original paper for your reference. 

| Model name | Paper link | Accuracy | Download link |
|------------|------------|----------|--------------|
| Vec2Face | https://arxiv.org/abs/2409.02979 | 92.00 | [GitHub](https://github.com/HaiyuWu/vec2face) |
| DCFace | https://arxiv.org/abs/2304.07060 | 89.56 | [GitHub](https://github.com/mk-minchul/dcface) |
| IDiff-Face | https://arxiv.org/abs/2308.04995 | 88.20 | [GitHub](https://github.com/fdbtrs/idiff-face) |
| GANDiffFace | https://arxiv.org/abs/2305.19962 | - | [GitHub](https://github.com/PietroMelzi/GANDiffFace) |
| DigiFace | https://arxiv.org/abs/2210.02579 | - | [GitHub](https://github.com/microsoft/DigiFace1M) |
| SFace | https://arxiv.org/abs/2206.10520 | 77.71 | [GitHub](https://github.com/fdbtrs/SFace-Privacy-friendly-and-Accurate-Face-Recognition-using-Synthetic-Data) |
| SynFace | https://arxiv.org/abs/2108.07960 | 74.75 | [GitHub](https://github.com/haibo-qiu/SynFace) |

Note that [CemiFace](https://github.com/szlbiubiubiu/CemiFace) can be a candidate, but you must drop the real identities out of the dataset first.
### Training sets

We support using .txt file to train the model. Using [file_path_extractor.py](./file_path_extractor.py) to get all the image paths and replacing the path in [./configs/arcface_r50_default.py](./configs/arcface_r50_default.py).
```bash
# usage of file_path_extractor.py
python3 file_path_extractor.py \
-s parent/folder/path/of/your/dataset \
-d destination \
-sfn filename \
-end_with jpg
# replacement at arcface_r50_default.py
config.train_source = "path/to/.txt/file"
```
**Please ensure that only image files are inside the resource folder when using file_path_extract.py.**
### Test sets
For your convenience, some test sets commonly used test sets, *e.g.*, LFW, CFP-FP, CALFW, CPLFW, AgeDB-30, can be downloaded [here](https://drive.google.com/file/d/1l7XmqzIZKdKVqu0cOS2EI0bL_9_-wIrc/view?usp=drive_link).
Extract the compressed file then you can simply run [prepare_test_images.py](https://github.com/HaiyuWu/SOTA-FR-train-and-test/blob/main/utils/prepare_test_images.py) to get datasets ready to test
```
python3 utils/prepare_test_images.py \
--xz_folder folder/contains/xz/files \
--destination ./test_set_package_5 \
--datasets lfw cfp_fp agedb_30 calfw cplfw
```
If you use a different destination, please change the corresponding configuration in [./configs/arcface_r50_default.py](./configs/arcface_r50_default.py).
## FR model training
Training command line with 4 GPUs:
```
torchrun --nproc_per_node=4 train.py --config_file ./configs/arcface_r50_default.py
```
### ❗Note that, you are not supposed to change any training hyperparameters❗

### Acknowledgement
The code is mainly based on [SOTA-Face-Recognition-Train-and-Test](https://github.com/HaiyuWu/SOTA-Face-Recognition-Train-and-Test)

### Q&A
1. Do I have to submit in all scales? \
Answer: No, choosing one or more scales is fine.
2. Can I use the available synthetic training sets? \
Answer: Yes, any of the available synthetic training sets are ok to be used.
3. What is the prize of the winner? \
Answer: The top two teams that achieve the highest accuracy and the teams that achieve higher accuracy than the baseline can directly publish their writeup at the workshop.

If you have other questions, please feel free to contact me at haiyupersonal@gmail.com

## License
[MIT license](./license.md)

