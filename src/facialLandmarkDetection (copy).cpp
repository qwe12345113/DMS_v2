#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <bits/stdc++.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "lib/drawLandmarks.hpp"
#include "lib/utils_math.hpp"
#include "lib/Eye_Dector_Module.hpp"
#include "lib/facerec.h"

using namespace dlib;
using namespace std;
// using namespace config;
//  using namespace cv;
//  ----------------------------------------------------------------------------------------
// #define EYE_AR_THRESH 0.25
// #define MOUTH_AR_THRESH 0.6
#define HEAD_X_THRESH 50
#define HEAD_Y_THRESH 35
#define YAWN_FRAME 15
#define EYE_AR_SLEEP_FRAME 10
#define LOWER_HEAD_FRAME 15
#define TURN_AROUND_FRAME 15
#define LAG 30


bool dirExists(const std::string &path)
{
  struct stat info;
  if (stat(path.c_str(), &info) == 0 && info.st_mode & S_IFDIR)
  {
    return true;
  }
  else{
    if (mkdir(path.c_str(), 0777) == -1)
      cerr << "Error :  " << strerror(errno) << endl;
  
    else
      cout << "Directory created" << endl;
    return false;
  }
}

int main(int argc, char **argv)
{
  cout << argc << endl;
  // init_config();
  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor sp;
  deserialize("../shape_predictor_68_face_landmarks.dat") >> sp;

  std::vector<full_object_detection> shapes;
  string command = argv[1], user_name = argv[2];
  string user_dir = "../database/" + user_name + "/";
  if (command == "pic")
  {
    image_window win;
    clock_t start = clock();
    cv::Mat frame = cv::imread(argv[2]);
    array2d<rgb_pixel> img;
    assign_image(img, cv_image<bgr_pixel>(frame));

    // Make the image larger so we can detect small faces.
    // pyramid_up(img);

    std::vector<full_object_detection> shapes;
    shapes = process(img, sp, detector);
    cout << eye_aspect_ratio(shapes.at(0)) << endl;

    cout << ((double)(clock() - start)) * 1000 / CLOCKS_PER_SEC << endl;
    // Now let's view our face poses on the screen.
    win.clear_overlay();
    win.set_image(img);
    win.add_overlay(render_face_detections(shapes));
    win.wait_until_closed();
  }
  else
  {
    //------- init ----------//
    cv::VideoCapture cam;
    if (command == "cam")
    {
      cout << "open camera " << endl;
      cam.open(0);
    }
    else if (command == "video")
    {
      cout << "read video " << endl;
      cam.open(argv[2]);
    }
    else
    {
      cout << "wrong command" << endl;
      return 1;
    }
    if (!cam.isOpened())
    {
      cout << "Could not open video or camera source\n";
      return 1;
    }

    cv::Mat frame;
    cv::Scalar color(0, 0, 255);

    int colse_eye_frame = 0, yawn_frame = 0, lower_head_frame = 0, trun_around_frame = 0, lag = 0, capture=1, n=0;
    bool yawn = false, recognized = 0, find_normal_satus_OK = 0, registered = dirExists(user_dir);
    float eye_ear = 0, mouth_ear = 0;
    std::vector<full_object_detection> shapes, tmp_shapes;
    std::vector<float> threshold;

    cv::VideoWriter writer;
    writer.open("./demo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 15.0, cv::Size(640, 360), true);

    EyeDetector Eye_det;

    /// start to capture frames from camera or video ///
    while (cam.read(frame))
    {
      // clock_t start = clock();
      cv::resize(frame, frame, cv::Size(640, 360));

      cv::Mat gray;
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

      // if the img is gray scale concat image
      // three time to simulate the color image
      if (frame.channels() == 1)
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);

      //------------ opencv format to dlib format ---------------//
      array2d<rgb_pixel> img;
      assign_image(img, cv_image<bgr_pixel>(frame));

      //----- Number of faces detected -----//
      std::vector<dlib::rectangle> dets = detector(img);

      if (dets.size() == 1)
      {

        full_object_detection landmarks = sp(img, dets[0]); // get face landmark
        eye_ear = eye_aspect_ratio(landmarks);
        mouth_ear = mouth_aspect_ratio(landmarks);
        shapes.push_back(landmarks);

        ///-------do face recognition first---------/////
        if (shapes.size() == 1 && !(recognized))
        {
          if (!registered)
          {
            cout << "user not exist, registering....." << endl;
            // for(int i = 0; i < 15; i++) // register new user
            // {                       
              //-------- save face image -------//
              if (capture && n == 15)
              {
                cv::Rect roi(dets[0].left(), dets[0].top(), dets[0].width(), dets[0].height());
                cv::Mat face = frame(roi);
                cv::cvtColor(face, face, cv::COLOR_BGR2GRAY);
                string save = user_dir + to_string(n) + ".pgm";
                cv::imwrite(save, face);
                capture = 0;
              }
            // }
            
          }
          cout << "start to recognize....." << endl;
          // checkFaceRecognition();
          // face recognition function
          // if success
          cout << "hellow XXX" << endl;
          recognized = 0;
          // else
          //   cout << "Account not found" << endl;
        }


        // start detect
        else if (shapes.size() == 1 && recognized)
        {
          cout << "detect" << endl;
        }
      }      

      // cout << ((double)(clock() - start)) * 1000 / CLOCKS_PER_SEC << " S" << endl;

      writer.write(frame);
      cv::imshow("123", frame);
      char key = cv::waitKey(1);
      if (key == 13){
        cout << "capture.";
        capture = capture ^ 1;
        n++;
      }
      shapes.clear();
    }
    cam.release();
    cv::destroyAllWindows();
  }
}
