# DMS_v1
## Introdution
This project is edit from [opencv-facial-landmark-detection](https://learnopencv.com/facemark-facial-landmark-detection-using-opencv/), but we change the **Facemark** model from opencv to dlib.

And we use the face landmark to detect the 4 of driving behaviors, including **yawn**, **distraction**, **lower head**, and **closing eyes**.

## Requirements
[dlib](http://dlib.net/) 19.24

## model
The 68-Dlib's point model can be download from [here](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2).

## Preparing

Structure of this project should be：
```
├─ DMS_v1
     ├─ src
     ├─ dlib-19.24
     └─ shape_predictor_68_face_landmarks.dat
```

## Compile
    $ mkdir build && cd build && cmake .. && make
