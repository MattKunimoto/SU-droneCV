//Threaded capture and retrieval
//Display in window
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <raspicam/raspicam_cv.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
using namespace std; 
using namespace cv;
using namespace raspicam;

mutex mtx;
condition_variable convar;
bool run = false;
bool runCap = false;
bool grabbable = false;
bool writeable = false;

void capture(RaspiCam_Cv& camera)
{
	unique_lock<mutex> lk(mtx);
	camera.open();
	camera.grab();
	run = true;
	convar.notify_one();
	lk.unlock();
}

void display(RaspiCam_Cv& camera)
{
	unique_lock<mutex> lk(mtx);
	Mat image;
	image.create(1280,960,CV_8UC1);
	namedWindow("image", WINDOW_AUTOSIZE);
	convar.wait(lk, []{return run;});
	camera.retrieve(image);
	imshow("image", image);
	lk.unlock();
	waitKey();
}

int main()
{
	Mat image;
	RaspiCam_Cv camera;
	camera.set (CV_CAP_PROP_FORMAT, CV_8UC1);
	thread cameraCapture(capture, ref(camera));
	thread imageDisplay(display, ref(camera));
	cameraCapture.join();
	imageDisplay.join();
}
