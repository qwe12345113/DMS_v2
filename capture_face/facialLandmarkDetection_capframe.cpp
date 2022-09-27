// Reference:
//		[1]Tutorial: https://www.learnopencv.com/facemark-facial-landmark-detection-using-opencv/
//		[2]Code: https://github.com/spmallick/learnopencv/tree/master/FacialLandmarkDetection
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>
#include "drawLandmarks.hpp"
#include <ctime>
#include <bits/stdc++.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;
using namespace cv;
using namespace cv::face;

int main(int argc, char **argv)
{
  CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");
  Ptr<Facemark> facemark = FacemarkLBF::create();
  facemark->loadModel("lbfmodel.yaml");

  string command = argv[1];

  Mat frame;

  VideoCapture cam;
  cam.open(0);
  int c =0;
  string pic_dir = "./"+ command + "/jpg/";
  string pic_dir2 = "./"+ command + "/pgm/";
  if (mkdir(command.c_str(), 0777) == -1)
        cerr << "Error :  " << strerror(errno) << endl;
  
  else
      cout << "Directory created" << endl;
  
  if (mkdir(pic_dir.c_str(), 0777) == -1)
        cerr << "Error :  " << strerror(errno) << endl;
  
  else
      cout << "Directory created" << endl;
  if (mkdir(pic_dir2.c_str(), 0777) == -1)
        cerr << "Error :  " << strerror(errno) << endl;
  
  else
      cout << "Directory created" << endl;
  
  while (cam.read(frame))
  {
    Mat gray, out;
    vector<Rect> faces;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    faceDetector.detectMultiScale(gray, faces);
    // cv::Rect rect(0, 0, im.size().width, im.size().height);
    // faces.emplace_back(rect);
    // cout << faces.size() << endl;
    if (faces.size() > 0)
    {
      // cout << faces.at(0).x << endl;
      // cout << faces.at(0).y << endl;
      // cout << faces.at(0).width << endl;
      // cout << faces.at(0).height << endl;

      cv::Rect myROI(faces.at(0).x, faces.at(0).y, faces.at(0).width, faces.at(0).height);
      out = gray(myROI);

      cv::resize(out, out, cv::Size(92, 112));
      string s = pic_dir + to_string(c) + ".jpg";
      string s2 = pic_dir2 + to_string(c) + ".pgm";
      imwrite(s,out);
      imwrite(s2,out);
      c++;
      imshow("face", out);
      if (waitKey(1) == 27)
        break;
    }
    else{
      imshow("Facial Landmark Detection", frame);
      if (waitKey(1) == 27)
        break;
    }
  }
  
  cam.release();

  cv::destroyAllWindows();
  return 0;
}
