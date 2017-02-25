//Testing tolerance of camera and Mat via pass by reference within refWrapper

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <raspicam/raspicam_cv.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

using namespace std;
using namespace cv;
using namespace raspicam;
using namespace chrono;

mutex bufferMtx;
condition_variable bufferCV;
bool bufferReading = false;
bool bufferWriting = true;

void grabAndRetrieveImage(RaspiCam_Cv& camera, Mat& buffer)
{
	camera.grab();
	camera.retrieve(buffer);
	imwrite("img1.bmp", buffer);
	camera.grab();
	camera.retrieve(buffer);
	imwrite("img2.bmp", buffer);
	camera.grab();
	camera.retrieve(buffer);
}

int main(int argc, char** argv)
{
	RaspiCam_Cv camera;
	Mat img;
	camera.set(CV_CAP_PROP_FORMAT, CV_8UC1);
	camera.open();
	thread cam(grabAndRetrieveImage, ref(camera), ref(img));
	cam.join();
	imwrite("img3.bmp", img);
	return 0;
}
