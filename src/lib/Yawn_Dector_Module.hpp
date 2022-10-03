#include <iostream>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace dlib;

class YwanDetector
{
public:
    full_object_detection keypoints;
    Mat frame;

    float get_EAR(Mat in_frame, full_object_detection landmarks)
    {
        keypoints = landmarks;
        frame = in_frame;
        std::vector<point> mouth_pts;

        int pts1[4] = {48, 50, 51, 52};
        int pts2[4] = {54, 56, 57, 58};

        for (int n = 0; n < 4; n++)
        {
            mouth_pts.push_back(point(pts1.part(pts1[n]).x(), pts1.part(pts1[n]).y()));
            mouth_pts.push_back(point(pts2.part(pts2[n]).x(), pts1.part(pts2[n]).y()));
        }

        float ear_mouth = EAR_mouth(mouth_pts);

        return ear_mouth;
    }

private:
    
    float LA_norm(point &p1, point &p2)
    {
        float x = p1.x() - p2.x();
        float y = p1.y() - p2.y();
        return sqrt(pow(x, 2) + pow(y, 2));
    }

    float EAR_mouth(std::vector<point> &eye_pts)
    {
        float a = (LA_norm(mouth_pts.at(0), mouth_pts.at(1));
        float b = (LA_norm(mouth_pts.at(2), mouth_pts.at(3));
        float c = (LA_norm(mouth_pts.at(4), mouth_pts.at(5));
        float d = (LA_norm(mouth_pts.at(6), mouth_pts.at(7));
        
        float mouth_ear = (a + b + c) / (3 * d);
        return ;
    }
};