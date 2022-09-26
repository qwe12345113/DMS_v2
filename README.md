# DMS_v1
## Introdution
This project is edit from [opencv-facial-landmark-detection](https://learnopencv.com/facemark-facial-landmark-detection-using-opencv/), but we change the **Facemark** model from opencv to dlib.

And we use the face landmark to detect the 4 of driving behaviors, including **yawn**, **distraction**, **lower head**, and **closing eyes**.

## Requirements
[dlib](http://dlib.net/)  == 19.24

opencv >= 3.4.14

## Model
You can download the 68-Dlib's point model from [here](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2).

## Preparing
Extract the dlib and model, and place them in to DMS_v1.

Structure of this project should be：
```
├─ DMS_v1
     ├─ src
     ├─ dlib-19.24
     └─ shape_predictor_68_face_landmarks.dat
```

## Compile
    $ mkdir build && cd build && cmake .. && make

## Getting Started:
### Usage
* detect image
```bash
$ ofld --pic /path/to/image
```
* detect video
```bash
$ ofld --video /path/to/video
```
* real-time detect 
```bash
$ ofld --cam
```
