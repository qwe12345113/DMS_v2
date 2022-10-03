# DMS_v2
## Introdution
In this project, we add the face recognition function into DMS_v1, but only use in detecting from webcam.

We use the face landmarks to detect the 4 of driving behaviors, including **yawn**, **distraction**, **lower head**, and **closing eyes**.

## Requirements
- gcc >= 4.7
- cmake >= 3.1
- opencv >= 3.4.14
- dlib == 19.24

## Preparing
1. You can download **dlib** from [here](http://dlib.net/), and **68-Dlib's point model** from [here](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2).

2. Extract the dlib and model, place them into DMS_v1. 
3. Structure of this project should be：
```
DMS_v2
  ├─ src
  ├─ dlib-19.24
  └─ Model
      └─ shape_predictor_68_face_landmarks.dat
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
# Demo
https://user-images.githubusercontent.com/11375811/192212654-d54cafaa-fb20-4ba7-aae6-e89cd07a4d1a.mp4

# Reference
[1] Tutorial 1：<https://www.learnopencv.com/facemark-facial-landmark-detection-using-opencv/>

[2] Tutorial 2：<https://github.com/spmallick/learnopencv/tree/master/FacialLandmarkDetection>

[3] Tutorial 3：<https://blog.csdn.net/u013841196/article/details/85041007>

[4] Yawn Detection：<https://github.com/deveshdatwani/facial-expression-recognition>

[5] Drowsiness detection：<https://pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/>

[6] Real-Time Eye Blink Detection using Facial Landmarks：<http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf>
