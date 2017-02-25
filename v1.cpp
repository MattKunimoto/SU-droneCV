//New version of main code
//Has thread for capturing image to buffer
//Has classification manager integrated into the main loop
//Has classifier threads

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <raspicam/raspicam_cv.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <string>

using namespace std;
using namespace cv;
using namespace raspicam;

static const char BACK = 'b';
static const char FRONT = 'f';
static const char SIDE = 's';

static const string backCascadeName = "/home/pi/seniordesign/classifiers/back/cascade.xml";
static const string frontCascadeName = "/home/pi/seniordesign/classifiers/front/cascade.xml";
static const string sideCascadeName = "/home/pi/seniordesign/classifiers/side/cascade.xml";

mutex bufferMtx;
mutex classifierMtx;
condition_variable bufferCV;
condition_variable classifierCV;
bool bufferWriting = false;
bool bufferReading = false;
bool running = false;
bool classificationDone = false;
bool startClassifiers = false;

void grabAndRetrieveImage(RaspiCam_Cv& camera, Mat& buffer)
{
	unique_lock<mutex> lk(bufferMtx, defer_lock);
	while(running){
		lk.lock();
		bufferCV.wait(lk, []{return !bufferReading;});
		bufferWriting = true;
		camera.grab();
		camera.retrieve(buffer);
		bufferWriting = false;
		lk.unlock();
		bufferCV.notify_one();
	}
}

void classifier(CascadeClassifier cascade, Mat& img, static const char side)
{
	unique_lock<mutex> lk(classifierMtx, defer_lock);
	while(running){
		lk.lock(); //this lock needs to be sharable between 3 threads I think but not sure
		classifierCV.wait(lk, []{return startClassifiers;});
		lk.unlock();
		cascade.detectMultiScale(buffer, detectionsRect, 1.1, 2, 0, Size(10, 10));
	}
}

void classificationManager(Mat& buffer)
{
	//Gets image from buffer
	Mat classifierImg;
	CascadeClassifier backCascade;
	//CascadeClassifier frontCascade;
	//CascadeClassifier sideCascade;
	
	if(!backCascade.load(backCascadeName) cout << "Error couldn't load back cascade" << endl;
	//if(!frontCascade.load(frontCascadeName) cout << "Error couldn't load front cascade" << endl;
	//if(!sideCascade.load(sideCascadeName) cout << "Error couldn't load side cascade" << endl;
	
	thread backClassifier(classifier, backCascade, ref(classifierImg), BACK);
	thread frontClassifier(classifier, frontCascade, ref(classifierImg), FRONT);	
	thread sideClassifier(classifier, sideCascade, ref(classifierImg), SIDE);
	
	unique_lock<mutex> lk(bufferMtx, defer_lock);
	while(running){
		lk.lock();
		bufferCV.wait(lk, []{return !bufferWriting;});
		bufferReading = true;
		classifierImg = buffer;
		bufferReading = false;
		lk.unlock();
		bufferCV.notify_one();
	}
}

int main(int argc, char** argv)
{
	//Initialize camera and current image and classification image
	RaspiCam_Cv camera;
	Mat cameraImg;
	
	//Set up and open camera
	camera.set(CV_CAP_PROP_FORMAT, CV_8UC1);
	camera.open();
	
	//Start capturing current images
	thread camFeed(grabAndRetrieveImage, ref(camera), ref(cameraImg));
	thread classMngr(classificationManager, ref(cameraImg));
	
	running = true;
	
	
}
