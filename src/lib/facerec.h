#ifndef FACE_REC_FACEREC_H
#define FACE_REC_FACEREC_H

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <array>
#include <ctime>
#include <string.h>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>

#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

#define INTERNAL 1
#define EXTERNAL 2
#define BEST_THRESHOLD 0.343
using namespace dlib;
using namespace std;

string checkFaceRecognition(string filename, string avoid);
void loopFaceRecognition(int type);

#endif //FACE_REC_FACEREC_H

