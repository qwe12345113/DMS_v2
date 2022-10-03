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
#include "lib/Register.hpp"

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

void DMS_pic(string img_path, frontal_face_detector detector, shape_predictor sp)
{
  cout << "read picture " << endl;
  image_window win;
  clock_t start = clock();
  cv::Mat frame = cv::imread(img_path.c_str());
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

void DMS_video(cv::VideoCapture cam, frontal_face_detector detector, shape_predictor sp)
{
  cout << "read video " << endl;
  cv::Mat frame;
  cv::Scalar color(0, 0, 255);

  int colse_eye_frame = 0, yawn_frame = 0, lower_head_frame = 0, trun_around_frame = 0, lag = 0;
  bool yawn = false, find_normal_satus_OK = 0;
  float eye_ear = 0, mouth_ear = 0;
  std::vector<full_object_detection> shapes, tmp_shapes;
  std::vector<float> threshold;

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
      // start detect
      if (shapes.size() == 1)
      {
        if (find_normal_satus_OK)
        {
          //--------- detect close eye -------------//
          if (eye_ear < threshold.at(0))
          {
            colse_eye_frame++;
            if (EYE_AR_SLEEP_FRAME < colse_eye_frame)
            {
              string s_ce = "Warning: Close Eye";
              cout << s_ce << endl;
              cv::putText(frame, s_ce, Point(0, 60), FONT_HERSHEY_SIMPLEX, 1, color, 2);
            }
          }
          else
          {
            colse_eye_frame = 0;
          }

          //--------------- detect yawn ----------------//
          if (mouth_ear > threshold.at(1))
          {
            yawn_frame++;
            if (yawn_frame > YAWN_FRAME)
            {
              yawn = true;
              if (yawn)
              {
                string s_y = "Warning: Yawn";
                cout << s_y << endl;
                cv::putText(frame, s_y, Point(0, 90), FONT_HERSHEY_SIMPLEX, 1, color, 2);
                yawn = false;
                yawn_frame = 0;
              }
            }
          }
          else
          {
            yawn_frame = 0;
          }

          //---------- Distraction (lower head) --------------//
          if (shapes.at(0).part(30).y() - threshold.at(3) > HEAD_Y_THRESH)
          {
            lower_head_frame++;
            if (lower_head_frame > LOWER_HEAD_FRAME)
            {
              string s_lh = "Warning: Lower head";
              cout << s_lh << endl;
              cv::putText(frame, s_lh, Point(0, 120), FONT_HERSHEY_SIMPLEX, 1, color, 2);
            }
          }
          else
          {
            lower_head_frame = 0;
          }

          //------------- Distraction (turn around head) -----------//
          if (abs(shapes.at(0).part(30).x() - threshold.at(2)) > HEAD_X_THRESH)
          {
            trun_around_frame++;
            if (trun_around_frame > TURN_AROUND_FRAME)
            {
              string s_d = "Warning: Distraction";
              cout << s_d << endl;
              cv::putText(frame, s_d, Point(0, 150), FONT_HERSHEY_SIMPLEX, 1, color, 2);
            }
          }
          else
          {
            trun_around_frame = 0;
          }
          // shapes.clear();
        }

        else
        {
          cout << "normal status calculating" << endl;
          if (lag > LAG)
          {
            threshold = threshold_calculate(tmp_shapes);
            cout << "find normal status finish" << endl;
            find_normal_satus_OK = true;
            tmp_shapes.clear();
          }
          else
          {
            tmp_shapes.push_back(shapes.at(0));
            lag++;
          }
        }
      }
    }
    // cout << ((double)(clock() - start)) * 1000 / CLOCKS_PER_SEC << " S" << endl;

    //---------opencv show --------------//
    // cv::Mat out = dlib::toMat(img);
    // cv::cvtColor(out,out, cv::COLOR_RGB2BGR);
    // std::vector<cv::Point2f> landmark;
    // for(int i = 0; i < shapes.at(0).num_parts(); i++)
    // {
    //   landmark.push_back(cv::Point2d(shapes.at(0).part(i).x(), shapes.at(0).part(i).y()));
    // }
    // drawLandmarks(out, landmark);
    cv::imshow("123", frame);
    char key = cv::waitKey(1);

    if (key == 27)
      break;

    //-------- dlib show ---------//
    // win.clear_overlay();
    // win.set_image(img);
    // win.add_overlay(render_face_detections(shapes));

    shapes.clear();
  }
  cam.release();
  cv::destroyAllWindows();
}

void DMS_registor(string user_name, cv::VideoCapture cam, frontal_face_detector detector, shape_predictor sp)
{  
  cout << "registor" << endl;
  cv::Mat frame;
  std::vector<full_object_detection> shapes;
  Register usr_reg;
  cout << "Press Enter to save a photo to build database " << endl;
  usr_reg.dirExists(user_name);
  
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
      shapes.push_back(landmarks);
    }
    else if(dets.size() == 0)
      cout << "no face detected" << endl;
    else
      cout << "too many faces" << endl;
    cv::imshow("123", frame);
    char key = cv::waitKey(1);

    if (key == 13 && dets.size() == 1)
    {
      if (!usr_reg.registered || !usr_reg.enough_photo)
      {
        usr_reg.registor(dets[0], frame); 
      }
      
    }

    else if (key == 27)
      break;
    
    shapes.clear();
  }
  cam.release();
  cv::destroyAllWindows();
}

void DMS_recognize(cv::VideoCapture cam, frontal_face_detector detector, shape_predictor sp)
{  
  cv::Mat frame;
  
  std::vector<full_object_detection> shapes;
  Register usr_reg;
  cout << "Press Enter take a photo to recognize user " << endl;
  
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
      shapes.push_back(landmarks);
    }
    else if(dets.size() == 0)
      cout << "no face detected" << endl;
    else
      cout << "too many faces" << endl;
    
    cv::imshow("123", frame);
    char key = cv::waitKey(1);

    if (key == 13 && dets.size() == 1)
    {
      usr_reg.TakePhoto(dets[0], frame);
      usr_reg.recognize_usr();
      if (usr_reg.recognized)
        break;
    }

    else if (key == 27)
      break;
    
    shapes.clear();
  }
  cam.release();
  cv::destroyAllWindows();
}

void DMS(cv::VideoCapture cam, frontal_face_detector detector, shape_predictor sp)
{
  cv::Mat frame;
  cv::Scalar color(0, 0, 255);

  int colse_eye_frame = 0, yawn_frame = 0, lower_head_frame = 0, trun_around_frame = 0, lag = 0;
  bool yawn = false, find_normal_satus_OK = 0;
  float eye_ear = 0, mouth_ear = 0;
  std::vector<full_object_detection> shapes, tmp_shapes;
  std::vector<float> threshold;

  cv::VideoWriter writer;
  writer.open("./demo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 15.0, cv::Size(640, 360), true);
  EyeDetector Eye_det;

  cout << "Press Enter to take a photo to recgonize driver" << endl;

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
      // start detect
      if (shapes.size() == 1)
      {
        cout << "start to detect" << endl;
        if (find_normal_satus_OK)
        {

          //--------- detect close eye -------------//
          // cout << (eye_ear-threshold.at(0))/threshold.at(4) << endl;
          // cout << eye_ear << endl;
          // cout << Eye_det.get_EAR(gray, landmarks) << endl;
          Eye_det.get_Gaze_Score(gray, landmarks);
          if (eye_ear < threshold.at(0))
          {
            colse_eye_frame++;
            if (EYE_AR_SLEEP_FRAME < colse_eye_frame)
            {
              string s_ce = "Warning: Close Eye";
              cout << s_ce << endl;
              cv::putText(frame, s_ce, Point(0, 60), FONT_HERSHEY_SIMPLEX, 1, color, 2);
            }
          }
          else
          {
            colse_eye_frame = 0;
          }

          //--------------- detect yawn ----------------//
          if (mouth_ear > threshold.at(1))
          {
            yawn_frame++;
            if (yawn_frame > YAWN_FRAME)
            {
              yawn = true;
              if (yawn)
              {
                string s_y = "Warning: Yawn";
                cout << s_y << endl;
                cv::putText(frame, s_y, Point(0, 90), FONT_HERSHEY_SIMPLEX, 1, color, 2);
                yawn = false;
                yawn_frame = 0;
              }
            }
          }
          else
          {
            yawn_frame = 0;
          }

          //---------- Distraction (lower head) --------------//
          if (shapes.at(0).part(30).y() - threshold.at(3) > HEAD_Y_THRESH)
          {
            lower_head_frame++;
            if (lower_head_frame > LOWER_HEAD_FRAME)
            {
              string s_lh = "Warning: Lower head";
              cout << s_lh << endl;
              cv::putText(frame, s_lh, Point(0, 120), FONT_HERSHEY_SIMPLEX, 1, color, 2);
            }
          }
          else
          {
            lower_head_frame = 0;
          }

          //------------- Distraction (turn around head) -----------//
          if (abs(shapes.at(0).part(30).x() - threshold.at(2)) > HEAD_X_THRESH)
          {
            trun_around_frame++;
            if (trun_around_frame > TURN_AROUND_FRAME)
            {
              string s_d = "Warning: Distraction";
              cout << s_d << endl;
              cv::putText(frame, s_d, Point(0, 150), FONT_HERSHEY_SIMPLEX, 1, color, 2);
            }
          }
          else
          {
            trun_around_frame = 0;
          }
          // shapes.clear();
        }
        else
        {
          cout << "normal status calculating" << endl;
          if (lag > LAG)
          {
            threshold = threshold_calculate(tmp_shapes);
            cout << "find normal status finish" << endl;
            find_normal_satus_OK = true;
            tmp_shapes.clear();
          }
          else
          {
            tmp_shapes.push_back(shapes.at(0));
            lag++;
          }
        }
      }
    }
    else if (dets.size() == 0)
      cout << "no face" << endl;

    writer.write(frame);
    cv::imshow("123", frame);
    char key = cv::waitKey(1);

    if (key == 27)
    {
      break;
    }
    shapes.clear();
  }
  cam.release();
  cv::destroyAllWindows();
}

int main(int argc, char **argv)
{
  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor sp;
  deserialize("../Model/shape_predictor_68_face_landmarks.dat") >> sp;

  if (argc == 3)
  {
    string command = argv[1];
    if (command == "pic") 
      DMS_pic(argv[2], detector, sp);
    
    else if (command == "video")
    {
      cv::VideoCapture cam(argv[2]);
      if (!cam.isOpened())
      {
        cout << "Could not open video or camera source\n";
        return 1;
      }
      DMS_video(cam, detector, sp);
    }
    else
      cout << "Wrong command" << endl;
  }

  else if (argc == 2)
  {    
    string command = argv[1];
    cv::VideoCapture cam;
    cam.open(0);
    if (!cam.isOpened())
    {
      cout << "Could not open video or camera source\n";
      return 1;
    }
    if (command == "rec")
      DMS_recognize(cam, detector, sp);
    else
      DMS_registor(argv[1], cam, detector, sp);
    // DMS_registor(detector, sp);
  }  
    
  
  else if (argc == 1)
  {   
    cout << "cam" << endl;
    cv::VideoCapture cam(0);
    if (!cam.isOpened())
    {
      cout << "Could not open video or camera source\n";
      return 1;
    }
    DMS(cam, detector, sp);
  }  

  else
  {
    cout << "Wrong command" << endl;
  }
  cout << "finish" <<endl;
}
