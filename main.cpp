//Version 1 Drone CV
//

#include </usr/local/include/opencv2/core/core.hpp>
#include </usr/local/include/opencv2/highgui/highgui.hpp>
#include </usr/local/include/opencv2/imgproc/imgproc.hpp>
#include "raspicam_cv"
#include "private/private_impl.h"
#include "scaler.h"
#include <stdio.h>
#include <thread>
#include <mutex>
#include <lock>
#include <condition_variable>
#include <vector>

//Declare chars for different sides
static const char BACK = 'b';
static const char SIDE = 's';
static const char FRONT = 'f';

//CascadeClassifiers
CascadeClassifier backCascade;
CascadeClassifier frontCascade;
CascadeClassifier sideCascade;

//Paths to classifiers
static const string backCascadeName = "/home/pi/seniordesign/classifiers/back/classifier.xml";
static const string frontCascadeName = "/home/pi/seniordesign/classifiers/front/classifier.xml";
static const string sideCascadeName = "/home/pi/seniordesign/classifiers/side/classifier.xml";

//Vectors that contain detection location info
std::vector<cv::Rect> backPat;
std::vector<cv::Rect> frontPat;
std::vector<cv::Rect> sidePat;

//Threading primitives
std::mutex bufferMtx;
std::condition_variable runClassifiers;
std::mutex classifierMtx;
std::condition_variable classifierFinished;


cv::Mat buffer;

raspicam::RaspiCam_Cv camera;

bool running;

class locSizeSide
{
public:
	locSizeSide::locSizeSide()
	{
		X = 0;
		Y = 0;
		W = 0;
		H = 0;
		Side = 0;
	}

	locSizeSide::locSizeSide(int x, int y, int w, int h, char side)
	{
		X = x;
		Y = y;
		W = w;
		H = h;
		Side = side;
	}
	int X, Y, W, H;
	char Side;
}

void cameraFeed(raspicam::RaspiCam_Cv& camera)
{
	//import camera feed and save image to sd card
	while(running){
	/*	bufferUse.lock();
		buffer = cv::imread("pic.jpg");
		bufferUse.unlock();
	*/
		camera.grab();
	}
}

void grabImageFromCamera()
{
	while(running){
		std::lock_guard<std::mutex> bufferLock(bufferMtx);
		camera.retrieve(buffer);
		equalizeHist(buffer, buffer);
		runClassifiers.notify_all();
	}
}

void frontClassifier(raspicam::RaspiCam_Cv& camera, cv::Mat& buffer, locSizeSide::locSizeSide& positionInfo)
{
	//swap image through pipe like object or locks and pass by reference
	/*cv::Mat img, gray, harris, harris_norm, harris_norm_scaled;
	bufferUse.lock();
	img = buffer;
	bufferUse.unlock();

	cv::cvtColor(img, gray, CV_BGR2GRAY);
	gray.convertTo(gray, CV_32F);

	//detection parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	//detecting corners
	cv::cornerHarris( gray, harris, blockSize, apertureSize, k);

	//normalizing
	cv::normalize( harris, harris_norm, 0 ,255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
	cv::convertScaleAbs( harris_norm, harris_norm_scaled );

	for ( int j = 0; j < harris_norm.rows; j++ )
		for( int i = 0; i < harris_norm.cols; i++ )
			if( (int) harris_norm.at<float>(j,i) > 200)
				cv::circle( img, cv::Point( i , j), 5, cv::Scalar(0), 2, 8, 0 );

	cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display Image", img);

	cv::waitKey(0);
	*/
	std::unique_lock<std::mutex> bufferLock(bufferMtx);
	runClassifiers.wait();
	frontCascade.detectMultiScale(buffer, backPat, 1.1, 2, 0, cv::Size(10, 10));
	for(int i = 0; i < backPat.size())

}

int main( void )
{
	// 1. initialize camera and create data containers
	// 2. call thread of cameraFeed
	// 3. call thread of harris corners

 //create camera
	//setup camera settings
	camera.set(CV_CAP_PROP_FORMAT, CV_8UC1);
	camera.open(1); //initialize camera and start capturing


	locSizeSide::locSizeSide skater;

	bool running = true;

	std::thread camFeed(cameraFeed, std::ref(camera), std::ref(buffer), running);
	std::thread bufferSwap(grabImageFromCamera, std::ref(camera), buffer);
	std::thread backClassifier(backClassifier, std::ref(buffer), skater);
	std::thread sideClassifier(sideClassifier, std::ref(buffer), skater);
	std::thread frontClassifier(frontClassifier, std::ref(buffer), skater);

	camFeed.join();
	harris.join();

	camera.release();
	return 0;
}
