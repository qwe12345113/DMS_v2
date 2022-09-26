# DMS_v1
## Introdution
This project is edit from [opencv-facial-landmark-detection](https://learnopencv.com/facemark-facial-landmark-detection-using-opencv/), but we change the **Facemark** model from opencv to dlib.

And we use the face landmark to detect the 4 of driving behaviors, including **yawn**, **distraction**, **lower head**, and **closing eyes**.

## Requirements
- gcc >= 4.7
- cmake >= 3.1
- opencv >= 3.4.14
- dlib == 19.24

You can download dlib from [here](http://dlib.net/).

## Model
You can download the 68-Dlib's point model from [here](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2).

## Preparing
1. You can download **dlib** from [here](http://dlib.net/), and **68-Dlib's point model** from [here](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2).

2. Extract the dlib and model, and place them in to DMS_v1. 
3. Structure of this project should be：
```
├─ DMS_v1
     ├─ src
     ├─ dlib-19.24
     └─ shape_predictor_68_face_landmarks.dat
```

## Compile
    $ mkdir build && cd build && cmake .. && make -j4

## Getting Started:
### Usage
* detect from image
```bash
$ ./ofld --pic /path/to/image
```
* detect from video
```bash
$ ./ofld --video /path/to/video
```
* detect from webcam
```bash
$ ./ofld --cam
```

# Reference
[1] Tutorial 1：<https://www.learnopencv.com/facemark-facial-landmark-detection-using-opencv/>

[2] Tutorial 2：<https://github.com/spmallick/learnopencv/tree/master/FacialLandmarkDetection>

[3] Yawn Detection：<https://github.com/deveshdatwani/facial-expression-recognition>
