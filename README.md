# ADSNet: Adaptation of Distinct Semantic for Uncertain Areas in Polyp Segmentation, BMVC 2023 

Official Tensorflow implementation of [ADSNet: Adaptation of Distinct Semantic for Uncertain Areas in Polyp Segmentation](https://proceedings.bmvc2023.org/806/) BMVC 2023. 
 <br>
<p>Chonnam National University</p>

<b>Author:</b> <b>Quang-Vinh Nguyen, Van-Thong Huynh, Soo-Hyung Kim</b>

In The 34th British Machine Vision Conference, 20th - 24th November 2023, Aberdeen, UK.

## Architecture

<p align="center">
<img src="architecture.png" width=100% height=40% 
class="center">
</p>

## Qualitative Results

<p align="center">
<img src="qualitative.png" width=100% height=40% 
class="center">
</p>

## Usage:
### Recommended environment:
```
Python 3.8
Tensorflow 2.12.0
```
Please use ```pip install -r requirements.txt``` to install the dependencies.

### Data preparation:
Download the training and testing datasets [Google Drive](https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/1BTgT27VxvOgKpHrigwm7Bw?) [code:sydz] and move them into 'polyp/' folder.

### Training:
For training, run ```python train.py ```
### Testing:

For Kvasir dataset testing, run ```python test.py --pretrain_path weight.hdf5 --test_name kvasir```

For CVC-Clinic dataset testing, run ```python test.py --pretrain_path weight.hdf5 --test_name clinic```

For CVC-ColonDB dataset testing, run ```python test.py --pretrain_path weight.hdf5 --test_name colon```

For ETIS dataset testing, run ```python test.py --pretrain_path weight.hdf5 --test_name etis```

For CVC-300 dataset testing, run ```python test.py --pretrain_path weight.hdf5 --test_name cvc300```

### Image-level Polyp Segmentation Compared Results:
We also provide some result of baseline methods, You could download from [Google Drive](https://drive.google.com/file/d/1xvjRl70pZbOO6wI5p94CSpZK2RAUnUnx/view?usp=sharing)/[Baidu Drive](https://pan.baidu.com/s/14HtaePQk46YFDH5jRQDhwQ?pwd=qw9i) [code:qw9i], including our results and that of compared models.

## :bookmark_tabs: Citation
```
@inproceedings{Nguyen_2023_BMVC,
author    = {Quang Vinh Nguyen and Van Thong Huynh and Soo-Hyung Kim},
title     = {Adaptation of Distinct Semantics for Uncertain Areas in Polyp Segmentation},
booktitle = {34th British Machine Vision Conference 2023, {BMVC} 2023, Aberdeen, UK, November 20-24, 2023},
publisher = {BMVA},
year      = {2023},
url       = {https://papers.bmvc2023.org/0806.pdf}
}
'''
