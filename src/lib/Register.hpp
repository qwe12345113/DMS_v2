#include <iostream>
#include <opencv2/opencv.hpp>
// #include <stdio.h>
// #include <stdlib.h>
// #include <ftw.h>
// #include <filesystem>
// #include <sys/types.h>
// #include <dirent.h>
#include "facerec.h"

using namespace cv;
using namespace std;
using namespace dlib;

class Register
{
public:
    bool registered = false;
    bool recognized = false;
    bool start = true;
    bool enough_photo = false;
    string usr;
    string usr_path;
    string database_path = "../database/";
    string tmp_image_path = "../tmp_user_image/";
    int n = 0;
    int photo_amount_need = 15;

    void dirExists(const std::string &usr_name)
    {
        dirInit();
        usr = usr_name;
        usr_path = database_path + usr_name + "/";
        struct stat info;
        if (stat(usr_path.c_str(), &info) == 0 && info.st_mode & S_IFDIR)
        {
            n = check_dir();
            // if (n <= photo_amount_need)
            //     registered = false;
            // else
            //     registered = true;
            registered = true;
        }
        else
        {
            if (mkdir(usr_path.c_str(), 0777) == -1)
                cerr << strerror(errno) << endl;
            else
                cout << "Directory created" << endl;
            registered = false;
        }

        registor_or_not();
    }

    void TakePhoto(dlib::rectangle dets, cv::Mat frame)
    {
        string save;
        cv::Rect roi = getROI(dets, frame.cols, frame.rows);
        cv::Mat face = frame(roi);
        save = tmp_image_path + "0.jpg";
        cv::imwrite(save, face);
        start = false;
        cout << "OK! start to recognize user." << endl;
    }

    void registor(dlib::rectangle dets, cv::Mat frame)
    {
        string save;

        if (!registered || n <= photo_amount_need)
        {
            if (n != photo_amount_need)
            {
                // cv::Rect roi(dets.left()-20, dets.top()-50, dets.width()+20, dets.height()+60);

                cv::Rect roi = getROI(dets, frame.cols, frame.rows);
                cv::Mat face = frame(roi);
                // cv::cvtColor(face, face, cv::COLOR_BGR2GRAY);
                save = usr_path + to_string(n) + ".jpg";
                cout << save << endl;
                cv::imwrite(save, face);
            }
            else
            {
                cout << "finish regitor" << endl;
                registered = true;
                enough_photo = true;
            }
            n++;
        }            
    }

    void recognize_usr()
    {
        string tmp = tmp_image_path + "0.jpg";
        string usr_rec = checkFaceRecognition(tmp, "123");

        if (usr_rec == "unknow")
        {
            string out = "You need to registor first.\n";
            cout << out;            
        }
        else if (usr_rec == "no faces")
        {
            string out = "take a photo and recognize again.\n";
            cout << out;            
        }
            
        else
        {
            string out = "Hellow " + usr_rec + "\n";
            cout << out;
            recognized = true;
        }
    }

private:
    void dirInit()
    {
        if (mkdir(database_path.c_str(), 0777) == -1)
            cerr << strerror(errno) << endl;
        else
            cout << "Directory created" << endl;

        if (mkdir(tmp_image_path.c_str(), 0777) == -1)
            cerr << strerror(errno) << endl;
        else
            cout << "Directory created" << endl;
    }

    void registor_or_not()
    {
        if (registered && n >= photo_amount_need){
            cout << "user name exist." << endl;
            enough_photo = true;
        }
        else if (registered && n < photo_amount_need)
            cout << "user name exist, but user's photos not enough." << endl;
        else
            cout << "user name not exist, registering....." << endl;
    }

    int check_dir()
    {
        // usr_path = database_path + usr_name + "/";
        DIR *dp;
        int i = 0;
        struct dirent *ep;
        dp = opendir(usr_path.c_str());

        if (dp != NULL)
        {
            while (ep = readdir(dp))
                i++;

            (void)closedir(dp);
        }
        else
            perror("Couldn't open the directory");
        string out = "There's " + to_string(i - 2) + " files in the " + usr_path + " directory.\n";
        cout << out;

        return i -2;
    }

    cv::Rect getROI(dlib::rectangle dets, int img_col, int img_row)
    {
        int left, top, w, h = 0;

        if (dets.left() - 20 < 0)
            left = 0;
        else
            left = dets.left() - 20;

        if (dets.top() - 50 < 0)
            top = 0;
        else
            top = dets.top() - 50;

        if ((int)(left + dets.width() + 30) > img_col)
            w = img_col - left;
        else
            w = dets.width() + 30;

        if ((int)(top + dets.height() + 60) > img_row)
            h = img_row - top;
        else
            h = dets.height() + 60;
        cv::Rect roi(left, top, w, h);
        // cv::Rect roi(dets.left()-20, dets.top()-50, dets.width()+20, dets.height()+60);
        return roi;
    }

    void rmFileLists(string path)
    {
        DIR *pDir;
        DIR *qDir;
        char *p;
        struct dirent *ptr;
        if (!(pDir = opendir(path.c_str())))
            return;
        // cout << "Folder : " << path << endl;
        while ((ptr = readdir(pDir)) != 0)
        {
            if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
            {
                string tempString = path + "/" + ptr->d_name;
                if (!(qDir = opendir(tempString.c_str())) && (p = strstr(ptr->d_name, ".jpg")))
                {
                    // files.push_back(ptr->d_name);
                    string file = path + ptr->d_name;
                    remove(file.c_str());
                }
                closedir(qDir);
            }
        }
        closedir(pDir);
        remove(path.c_str());
    }
};
