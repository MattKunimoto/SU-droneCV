//Capture and retrieve in one thread
//display via pass by reference in another thread

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
	unique_lock<mutex> lk(bufferMtx);
	camera.grab();
	camera.retrieve(buffer);
	imwrite("img1.bmp", buffer);
	bufferWriting = false;
	lk.unlock();
	bufferCV.notify_all();
	this_thread::yield();
	
	lk.lock();
	bufferCV.wait(lk, []{bufferWriting = !bufferReading; return !bufferReading;});
	camera.grab();
	camera.retrieve(buffer);
	imwrite("img3.bmp", buffer);
	bufferWriting = false;
	lk.unlock();
	bufferCV.notify_all();
}


void display(Mat& buffer)
{
	unique_lock<mutex> lk(bufferMtx);
	bufferCV.wait(lk, []{bufferReading = !bufferWriting; return !bufferWriting;});
	imwrite("img2.bmp", buffer);
	bufferReading = false;
	lk.unlock();
	bufferCV.notify_all();
	this_thread::yield();
	
	lk.lock();
	bufferCV.wait(lk, []{bufferReading = !bufferWriting; return !bufferWriting;});
	imwrite("img4.bmp", buffer);
	bufferReading = false;
	lk.unlock();
	bufferCV.notify_all();
	this_thread::yield();
}

int main()
{
	RaspiCam_Cv camera;
	Mat buffer(1280, 960, CV_8UC1);
	camera.set(CV_CAP_PROP_FORMAT, CV_8UC1);
	camera.open();
	thread cap(grabAndRetrieveImage, ref(camera), ref(buffer));
	thread disp(display, ref(buffer));
	cap.join();
	disp.join();
	return 0;
}
