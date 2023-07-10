# Video-Matting
In this project, a refining method for video matting is implemenetd.

## Installation

Install the requirements:

```bash
  $ pip install -r requirements.txt
```

## Generating Data
In this step, you need to compose the dataset for this project, in order to do this we use 2 available datasets for the task of video matting. we use the foreground frames of different videos and also a shuffle background for each frame and we compose our own dataset. after creating the datatset, we pass our dataset to an image matting netwrok to get the base predictions.

#### please follow this steps:
1.

```bash
 $ mkdir matting
 $ cd matting
 $ git clone "remote URL"
 $ git clone "remote URL"
 $ mkdir data
 $ cd data
```
2. Download the VideoMatte240K_JPEG_HD dataset from this [link](https://drive.google.com/file/d/1IUp_301x8BnPjE81QBzyLASn3ZSosUF6/view).
3. Download the Backgournd dataset from this [link](https://drive.google.com/file/d/1FqD-HfwXwbeTswQEIFaQkaVWUh_i6cSy/view). 
4. Extract the datasets you downloaded in the matting folder.
5. Download the pretrained models you need from this [link](https://drive.google.com/file/d/1NzEjOtC9GqHnnLJoYfAx-l1_B-kEjYnX/view?usp=share_link). 
6. Extract the pretrained models that you downloaded in the matting folder.
7. ```bash
 $ python data_generation.py
```

