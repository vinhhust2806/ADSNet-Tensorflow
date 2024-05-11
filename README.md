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
python 3.9.13
tensorflow-gpu 2.10.0
```
Please use ```pip install -r requirements.txt``` to install the dependencies.

### Data preparation:
Download the training and testing datasets [Google Drive](https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view?usp=sharing) and move them into 'polyp/' folder.

### Training:
For training, run ```python train.py --epoch $epoch --batch_sise #batch_size --lr $learning_rate --min_lr $min_learning_rate --val_name $kvasir/clinic/colon/etis/endoscene```

### Testing:
For testing, run ```python test.py --pretrain_path $pretrain_path --test_name $kvasir/clinic/colon/etis/endoscene```

### Visualize:
run ```python visualize.py --pretrain_path $pretrain_path --test_name $kvasir/clinic/colon/etis/endoscene```

### Polyp Segmentation Compared Results:
We also provide some result of baseline methods, You could download from [Google Drive](https://drive.google.com/file/d/1xvjRl70pZbOO6wI5p94CSpZK2RAUnUnx/view?usp=sharing), including results of compared models.

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
