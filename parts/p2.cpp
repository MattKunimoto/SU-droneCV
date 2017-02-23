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
#include <vector>
using namespace std; 

bool running;
raspicam::RaspiCam_Cv Camera;
std::mutex bufferMtx;
std::condition_variable cameraCV;

void cameraFeed() {
	cout << "Enter camera feed" << endl;
	int i = 0;
	cout << "1a" << endl;
	std::unique_lock<std::mutex> lk(bufferMtx);
	cout << "2a" << endl;
	Camera.grab();
	cout << "3a" << endl;
	lk.unlock();
	cout << "4a" << endl;
	//cameraCV.notify_all();
	cout << "5a" << endl;
	while(running){
		i++;
		if(i % 30 == 0) cout << "capturing " << i << endl;
		Camera.grab();
	}
}

void retrieveAndDisplay() {
	cout << "1b" << endl;
	cv::Mat image;
	std::unique_lock<std::mutex> lk(bufferMtx);
	cout << "2b" << endl;
	//cameraCV.wait(lk);
	cout << "3b" << endl;
	char c = (char)cv::waitKey(10);
	cout << "4b" << endl;
	int i = 0;
	while(running){
		i++;
		Camera.retrieve(image);
		cv::imshow("image", image);
		if(c == 27) running = false;
	}
	cout << "5b" << endl;
}

int main ( ) {

    Camera.set( CV_CAP_PROP_FORMAT, CV_8UC1 );
    //Open camera
    cout<<"Opening Camera..."<<endl;
    if (!Camera.open()) {cerr<<"Error opening the camera"<<endl;return -1;}
    else running = true;
    //Start capture
    std::thread camFeed(cameraFeed);
    std::thread retAndDisp(retrieveAndDisplay);
    
    camFeed.join();
    retAndDisp.join();
    
    Camera.release();
    
    //save image 
    cout<<"Image saved at raspicam_cv_image.jpg"<<endl;
    return 0;
}
