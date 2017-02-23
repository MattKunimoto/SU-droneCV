//Version 1 Drone CV
//
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <raspicam/raspicam_cv.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>

//Declare chars for different sides
static const char BACK = 'b';
static const char FRONT = 'f';
static const char SIDE = 's';

bool DEBUG;

//Paths to classifiers
static const std::string backCascadeName = "/home/pi/seniordesign/classifiers/back/classifier.xml";
static const std::string frontCascadeName = "/home/pi/seniordesign/classifiers/front/classifier.xml";
static const std::string sideCascadeName = "/home/pi/seniordesign/classifiers/side/classifier.xml";

//Vectors that contain detection location info


//Mutex for managing the camera's buffer
std::mutex bufferMtx;


cv::Mat buffer;

bool running;

//Wrapper for cv::Rect. Adds char to track sides.
struct locSizeSide
{
public:
	locSizeSide(cv::Rect rect, char side)
	{
		Rect = rect;
		Side = side;
	}
	cv::Rect Rect;
	char Side;
};


void cameraFeed(raspicam::RaspiCam_Cv& camera);
void classifier(cv::CascadeClassifier& cascade, raspicam::RaspiCam_Cv& camera, cv::Mat& buffer, std::vector<locSizeSide>& detections, char side);
void classifierManager(raspicam::RaspiCam_Cv camera);


int main( int argc, char** argv )
{
	// 1. initialize camera and create data containers
	// 2. call thread of cameraFeed
	// 3. call thread of harris corners

 //create camera
	//setup camera settings

	if(argc == 1) DEBUG = false;
	else if(argc == 2 && argv[1] == "1") DEBUG = true;
	else {std::cerr << "Only arugment is 1 to turn on debug."; return -1;}

	if(DEBUG == true) std::cout << "Setting up camera." << std::endl;

	raspicam::RaspiCam_Cv camera;
	camera.set(CV_CAP_PROP_FORMAT, CV_8UC1);
	camera.open(); //initialize camera 

	running = true;

	if(DEBUG == true) std::cout << "Spawning camera feed." << std::endl;
	std::thread camFeed(cameraFeed, std::ref(camera));
	if(DEBUG == true) std::cout << "Spawing classification manager." << std::endl;
	std::thread classificationManager(classificationManager, std::ref(camera));
	

	if(DEBUG == true) std::cout << "Exiting program." << std::endl;
	camFeed.join();
	classificationManager.join();

	camera.release();
	return 0;
}

void cameraFeed(raspicam::RaspiCam_Cv& camera)
{
	if(DEBUG == true) std::cout << "Creating camera feed lock for camera buffer." << std::endl;
	std::unique_lock<std::mutex> bufferUse(bufferMtx, std::defer_lock); //Create lock for grabbing images from camera
	if(DEBUG == true) std::cout << "Lock created." << std::endl;
	int i = 0;
	while(running){
		if(i % 100 && DEBUG == true) std::cout << "Locking camera buffer from camera feed." << std::endl;
		bufferUse.lock(); //Lock camera buffer
		if(i % 100 && DEBUG == true) std::cout << "Grabbing image for buffer." << std::endl;
		camera.grab(); //Grab image to buffer
		if(i % 100 && DEBUG == true) std::cout << "Unlocking camera buffer from camera feed." << std::endl;
		bufferUse.unlock(); //Unlock camera buffer
		
		//add wait key for exit
	}
}

void classifier(cv::CascadeClassifier& cascade, raspicam::RaspiCam_Cv& camera, cv::Mat& buffer, std::vector<locSizeSide>& detections, char side)
{
	std::vector<cv::Rect> detectionsRect;
	detections.clear();
	cascade.detectMultiScale(buffer, detectionsRect, 1.1, 2, 0, cv::Size(10, 10));
	for(int i = 0; i < detectionsRect.size(); i++){
		detections.emplace_back(detectionsRect.at(i), side);
	}
}

void classifierManager(raspicam::RaspiCam_Cv camera)
{
	//Make Classifiers
	if(DEBUG == true) std::cout << "Making classifiers." << std::endl;
	cv::CascadeClassifier backCascade;
	cv::CascadeClassifier frontCascade;
	cv::CascadeClassifier sideCascade;

	//Load Classifiers
	if(DEBUG == true) std::cout << "Loading classifiers." << std::endl;
	backCascade.load(backCascadeName);
	frontCascade.load(frontCascadeName);
	sideCascade.load(sideCascadeName);

	//Create vectors of detections
	if(DEBUG == true) std::cout << "Creating vectors for detections.";
	std::vector<locSizeSide> backDetections;
	std::vector<locSizeSide> frontDetections;
	std::vector<locSizeSide> sideDetections;
	std::vector<locSizeSide> detections;  //vector that will contain all 3 sets this might have to come out globally so the instructional program can access it. Or we will need to create a pipe of some sort to transfer the data.

	//current image
	if(DEBUG == true) std::cout << "Create buffer for image for classification." << std::endl;
	cv::Mat buffer;

	int i = 0;

	if(DEBUG == true) std::cout << "Entering detection loop." << std::endl;
	while(running){
		//Clear vectors with detections
		/*if(i % 100 && DEBUG == true) std::cout << "Clearing detection vectors." << std::endl;
		backDetections.clear();
		frontDetections.clear();
		sideDetections.clear();*/

		//Grab current image from buffer
		if(i % 100 && DEBUG == true) std::cout << "Locking camera buffer from classifier manager." << std::endl;
		std::unique_lock<std::mutex> bufferLock(bufferMtx);
		if(i % 100 && DEBUG == true) std::cout << "Grabbing classifer buffer from camera buffer." << std::endl;
		camera.retrieve(buffer);

		if(i % 100 && DEBUG == true) std::cout << "Normalizing image brightness." << std::endl;
		equalizeHist(buffer, buffer);

		//Spawn the three classifiers
		if(i % 100 && DEBUG == true) std::cout << "Spawing classifier threads." << std::endl;
		std::thread backClassifier(classifier, std::ref(backClassifier), std::ref(buffer), std::ref(backDetections), BACK);
		std::thread frontClassifier(classifier, std::ref(frontClassifier), std::ref(buffer), std::ref(frontDetections), FRONT);
		std::thread sideClassifier(classifier, std::ref(sideClassifier), std::ref(buffer), std::ref(sideDetections), SIDE);

		//Wait for them to end
		backClassifier.join();
		sideClassifier.join();
		frontClassifier.join();
		if(i % 100 && DEBUG == true) std::cout << "Classification finished." << std::endl;

		//Clear and concatonate the deteciton vectors
		if(i % 100 && DEBUG == true) std::cout << "Clearing and concatonating vectors." << std::endl;
		detections.clear();
		detections.insert(detections.end(), backDetections.begin(), backDetections.end());
		detections.insert(detections.end(), sideDetections.begin(), sideDetections.end());
		detections.insert(detections.end(), frontDetections.begin(), frontDetections.end());

		//Add stuff to output detection vector.
		i++;
	}
	if(DEBUG == true) std::cout << "Exited detection loop." << std::endl;
}
