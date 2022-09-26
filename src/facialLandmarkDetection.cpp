#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "lib/drawLandmarks.hpp"
#include "lib/utils_math.hpp"

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
/*
void init_config()
{
#define CFG(x) \
  x = Config.get(#x)

  const char *config_file = "config.cfg";
  ConfigParser Config(config_file);
  CFG(EYE_AR_THRESH);
  CFG(MOUTH_AR_THRESH);
  CFG(HEAD_X_THRESH);
  CFG(HEAD_Y_THRESH);
  CFG(EYE_AR_CONSEC_FRAME);
  CFG(EYE_AR_SLEEP_FRAME);
  CFG(LOWER_HEAD_FRAME);
  CFG(TURN_AROUND_FRAME);
#undef CFG
}*/

int main(int argc, char **argv)
{
  cout << argc << endl;
  // init_config();
  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor sp;
  deserialize("../shape_predictor_68_face_landmarks.dat") >> sp;

  image_window win;
  std::vector<full_object_detection> shapes;
  string command = argv[1];
  if (command == "pic")
  {
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

    int colse_eye_frame = 0, yawn_frame = 0, lower_head_frame = 0, trun_around_frame = 0, lag = 0;
    bool yawn = false, recognized = 0, find_normal_satus_OK = 0;
    float eye_ear = 0, mouth_ear = 0;
    std::vector<full_object_detection> shapes;
    std::vector<full_object_detection> tmp_shapes;
    std::vector<float> threshold;

    cv::VideoWriter writer;
    writer.open("./demo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0, cv::Size(640, 360), true);

    /// start to capture frames from camera or video ///
    while (cam.read(frame))
    {
      // clock_t start = clock();
      cv::resize(frame, frame, cv::Size(640, 360));

      // if the img is gray scale concat img
      // three time to simulate the color img
      if (frame.channels() == 1)
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);

      //------------ opencv format to dlib format ---------------//
      array2d<rgb_pixel> img;
      assign_image(img, cv_image<bgr_pixel>(frame));

      // Number of faces detected
      std::vector<dlib::rectangle> dets = detector(img);
      if (dets.size() == 1)
      {

        full_object_detection shape = sp(img, dets[0]); // get face landmark
        eye_ear = eye_aspect_ratio(shape);
        mouth_ear = mouth_aspect_ratio(shape);
        shapes.push_back(shape);

        ///-------do face recognition first---------/////
        if (shapes.size() == 1 && !(recognized))
        {
          // face recognition function
          // if success
          cout << "hellow XXX" << endl;
          recognized = 1;
          // else
          //   cout << "Account not found" << endl;
        }

        else if (shapes.size() == 1 && recognized)
        {
          if (find_normal_satus_OK)
          {
            
            //--------- detect close eye -------------//
            // cout << (eye_ear-threshold.at(0))/threshold.at(4) << endl;
            cout << eye_ear << endl;
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
      else // if can not find faceor or find more than one face, we determine this case as "Distraction".
      {
        string s_d = "Warning: Distraction";
        cout << s_d << endl;
        cv::putText(frame, s_d, Point(0, 150), FONT_HERSHEY_SIMPLEX, 1, color, 2);
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
      
      writer.write(frame);
      cv::imshow("123", frame);
      cv::waitKey(1);

      //-------- dlib show ---------//
      // win.clear_overlay();
      // win.set_image(img);
      // win.add_overlay(render_face_detections(shapes));

      shapes.clear();
    }
    cam.release();
    cv::destroyAllWindows();
  }
}
