# DMS_v2
## Introdution
In this project, we add the face recognition function into DMS_v1.

We use the face landmarks to detect the 4 of driving behaviors, including **yawn**, **distraction**, **lower head**, and **closing eyes**.

## Requirements
- gcc >= 4.7
- cmake >= 3.1
- opencv >= 3.4.14
- dlib == 19.24

## Preparing
1. Download [dlib](http://dlib.net/) and the following model：
    - [face_recognition_resnet_model](https://github.com/davisking/dlib-models/blob/master/dlib_face_recognition_resnet_model_v1.dat.bz2)
    - [68-Dlib's point model](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)
    - [5-Dlib's point model](https://github.com/davisking/dlib-models/blob/master/shape_predictor_5_face_landmarks.dat.bz2)
    

2. Extract the dlib and model, place them into DMS_v2. 
3. Structure of this project should be：
```
DMS_v2
  ├─ src
  ├─ dlib-19.24
  └─ Model
      ├─ shape_predictor_68_face_landmarks.dat
      ├─ shape_predictor_5_face_landmarks.dat
      └─ dlib_face_recognition_resnet_model_v1.dat
```

## Compile
    $ mkdir build && cd build && cmake .. && make -j4

## Getting Started:
### Usage
* detect from image.
```bash
$ ./dms pic /path/to/image
```
* detect from video.
```bash
$ ./dms video /path/to/video
```
* detect from webcam.
```bash
$ ./dms
```
* registor.
```bash
$ ./dms name
```
* face recognition.
```bash
$ ./dms rec
```


# Reference
[1] Tutorial 1：<https://www.learnopencv.com/facemark-facial-landmark-detection-using-opencv/>

[2] Tutorial 2：<https://github.com/spmallick/learnopencv/tree/master/FacialLandmarkDetection>

[3] Tutorial 3：<https://blog.csdn.net/u013841196/article/details/85041007>

[4] Yawn Detection：<https://github.com/deveshdatwani/facial-expression-recognition>

[5] Drowsiness detection：<https://pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/>

[6] Real-Time Eye Blink Detection using Facial Landmarks：<http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf>
