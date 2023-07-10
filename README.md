# Video-Matting
In this project, a refining method for video matting is implemenetd.

## Installation

Install the requirements:

```bash
  $ pip install -r requirements.txt
```

## Generating Data
In this step, you need to compose the dataset for this project, in order to do this we use 2 available datasets for the task of video matting. we use the foreground frames of different videos and also a shuffle background for each frame and we compose our own dataset.

#### please follow this steps:
1.

```bash
 $ mkdir matting
```
```bash
 $ cd matting
```
$ git clone "remote URL"
```
  ```bash
  $ git clone "remote URL"
  ```

6. Download the VideoMatte240K_JPEG_HD dataset from this [link](https://drive.google.com/file/d/1IUp_301x8BnPjE81QBzyLASn3ZSosUF6/view).
7. Download the Backgournd dataset from this [link](https://drive.google.com/file/d/1FqD-HfwXwbeTswQEIFaQkaVWUh_i6cSy/view).
  
8. Extract the datasets you downloaded in the matting folder.

9. Download the pretrained models you need from this [link](https://drive.google.com/file/d/1NzEjOtC9GqHnnLJoYfAx-l1_B-kEjYnX/view?usp=share_link).
  
10. Extract the pretrained models that you downloaded in the matting folder.
11.
```bash
  $ cd Video-Matting/video-segmentation/
```
```bash
  $ python negin_video.py
```

