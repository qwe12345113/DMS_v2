#include <iostream>
#include <ctime>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "lib/utils_math.hpp"
#include "lib/Eye_Dector_Module.hpp"
#include "lib/Yawn_Dector_Module.hpp"
#include "lib/Pose_Estimation_Module.hpp"
#include "lib/Register.hpp"
#include "lib/Attention_Scorer_Module.hpp"

using namespace dlib;
using namespace std;
using namespace cv;

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
  // clock_t start = clock();
  auto start = chrono::steady_clock::now();
  cv::Mat frame = cv::imread(img_path.c_str());
  array2d<rgb_pixel> img;
  assign_image(img, cv_image<bgr_pixel>(frame));

  // Make the image larger so we can detect small faces.
  // pyramid_up(img);

  std::vector<full_object_detection> shapes;
  shapes = process(img, sp, detector);
  cout << eye_aspect_ratio(shapes.at(0)) << endl;

  auto elasped = chrono::steady_clock::now() - start;
  auto sec_float = chrono::duration<float>(elasped); 
  cout << sec_float.count() << endl;
  // cout << ((double)(clock() - start)) * 1000 / CLOCKS_PER_SEC << endl;
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
  
  usr_reg.dirExists(user_name);

  if (!usr_reg.registered)
    cout << "Press Enter to save a photo to build database " << endl;
  
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

    if (usr_reg.registered && usr_reg.enough_photo)
        break;

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
      cout << "no face detected." << endl;
    else
      cout << "too many faces." << endl;
    
    cv::imshow("123", frame);
    char key = cv::waitKey(1);
    
    if (key == 13 && dets.size() == 1)
    {
      usr_reg.TakePhoto(dets[0], frame);
      usr_reg.recognize_usr();
      break;
    }
    // else if (key == 32)
    //   cout << "space\n";

    else if (key == 27)
      break;
    
    shapes.clear();
  }
  cam.release();
  cv::destroyAllWindows();
}

void DMS_old(cv::VideoCapture cam, frontal_face_detector detector, shape_predictor sp)
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
        if (find_normal_satus_OK)
        {

          //--------- detect close eye -------------//
          // cout << (eye_ear-threshold.at(0))/threshold.at(4) << endl;
          // cout << eye_ear << endl;
          // cout << Eye_det.get_EAR(gray, landmarks) << endl;
          
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

void DMS(cv::VideoCapture cam, frontal_face_detector detector, shape_predictor sp)
{
  cv::Mat frame;
  cv::Scalar color(0, 0, 255);

  int lag = 0, fps_lim = 12;
  // float time_lim = 1. / fps_lim ;
  bool find_normal_satus_OK = false, record=true, show_detail=false;

  std::vector<full_object_detection> shapes, tmp_shapes;
  std::vector<float> threshold;

  cv::VideoWriter writer;
  if(record)  
    writer.open("./demo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 15.0, cv::Size(640, 360), true);
  
  EyeDetector Eye_det;
  HeadPoseEstimator Head_pose;
  YawnDetector yawn_det;
  AttentionScorer Scorer;
  // Scorer.init(fps_lim, 0.26, 2, 0.2, 2, 35, 28, 2.5);
  
  string out="";
  float ear=0, m_ear=0, gaze=0, avg_pitch = 0;  
  
  while (cam.read(frame))
  {
    // clock_t start(clock());
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
      // start detect
      if (shapes.size() == 1)
      {
        if (find_normal_satus_OK)
        {
          ear = Eye_det.get_EAR(frame, landmarks);
          Scorer.get_PERCLOS(ear); // get the tired and perclos_score

          m_ear = yawn_det.get_EAR(frame, landmarks); // get mouth EAR

          Head_pose.get_pose(frame, landmarks); // frame, roll, pitch, yaw

          if(show_detail){
            out = "EAR: " + to_string(ear);
            cv::putText(frame, out, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 1);

            out = "Gaze Score: " + to_string(gaze);
            cv::putText(frame, out, Point(10, 80), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 1);

            out = "PERCLOS: " + to_string(Scorer.perclos_score);
            cv::putText(frame, out, Point(10, 110), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 1);

            out = "MEAR: " + to_string(m_ear);
            cv::putText(frame, out, Point(10, 130), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 1);
          }

          Scorer.eval_scores(ear, m_ear, Head_pose.pitch, Head_pose.yaw, shapes.at(0).part(30).y());

          if(Scorer.is_tired)
            cv::putText(frame, "Tired !", Point(10, 280), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 1);
          if(Scorer.is_asleep)
            cv::putText(frame, "CLOSE EYES !", Point(10, 300), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 1);
          if(Scorer.is_yawn)
            cv::putText(frame, "YAWN !", Point(10, 320), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 1);
          // if(Scorer.is_looking_away)
          //   cv::putText(frame, "LOOKING AWAY!", Point(400, 300), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 1);
          if(Scorer.is_lower_head)
            cv::putText(frame, "LOWER HEAD !", Point(400, 300), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 1);
          if(Scorer.is_distracted)
            cv::putText(frame, "DISTRACTED !", Point(400, 320), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 1);

        }
        else
        {
          cout << "normal status calculating" << endl;
          if (lag > LAG)
          {
            threshold = threshold_calculate(tmp_shapes);
            avg_pitch = avg_pitch / lag;
            cout << avg_pitch << endl;
            cout << "calculate finish" << endl;
            find_normal_satus_OK = true;
            tmp_shapes.clear();
            Scorer.init(fps_lim, threshold.at(0), 3, 0.2, 3, 28, 0.1, 2.5, threshold.at(1), 2, 1, threshold.at(3), avg_pitch);
            Head_pose.pitch = 0; // init pitch
            
          }
          else
          {
            tmp_shapes.push_back(shapes.at(0));
            Head_pose.get_pose(frame, landmarks);
            avg_pitch = avg_pitch + abs(Head_pose.pitch);
            lag++;
          }
        }
      }
    }
    else if (dets.size() == 0){
      // cout << "no face" << endl;
      cv::putText(frame, "No Face!", Point(200, 320), FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0,0,255), 2);
    }
    // double duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
    
    
    if(record)
      writer.write(frame);

    cv::imshow("123", frame);
    // cout << duration << endl;
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
      DMS_recognize(cam, detector, sp); // recognize
    else
      DMS_registor(argv[1], cam, detector, sp); // registor
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
