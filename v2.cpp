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
#include <atomic>
#include <chrono>

using namespace std;
using namespace cv;
using namespace raspicam;
using namespace chrono;

class RingBuffer{
	public:	
	//Creates triple buffer
	RingBuffer(){
		Node * ptr = nullptr;
		currentNode = new Node; //current is buff1
		ptr = new Node; //ptr is buff2
		currentNode->next = ptr; //buff1->next is buff2
		ptr->prev = currentNode; //buff2->prev is buff1
		ptr = new Node; //ptr is buff3
		currentNode->next->next = ptr; //buff2->next is buff3
		ptr->prev = currentNode->next; //buff3->prev is buff2
		currentNode->prev = ptr; //buff1->prev is buff3
		ptr->next = currentNode; //buff3->next is buff1
	}
	
	//overloaded assignement operator to add images
	void operator=(const Mat& image){
		currentNode->img = image;
		currentNode = currentNode->next;
	}
	
	~RingBuffer(){
		delete currentNode->next;
		delete currentNode->prev;
		delete currentNode;
	}
	
	Mat getLast(){
		return currentNode->prev->img;
	}
	
	private:
	class Node{
	public:
		Node(){
			img = Mat::zeros(1280, 960, CV_8UC1);
			next = nullptr;
			prev = nullptr;
		}
		
		Mat img;
		Node* next;
		Node* prev;
	};
	
	Node * currentNode = nullptr;
};

mutex mtx;
condition_variable bufCV;
bool buffersReady = false;
bool running = true;

void grabAndRetrieveImage(RaspiCam_Cv& camera, RingBuffer& buffer)
{
	unique_lock<mutex> lk(mtx);
	
	Mat img;
	
	for(int i = 0; i < 3; i++){
		camera.grab();
		camera.retrieve(img);
		buffer = img;
	}
	buffersReady = true;
	lk.unlock();
	bufCV.notify_one();
	
	while(running){
		camera.grab();
		camera.retrieve(img);
		buffer = img;
	}
}


void display(RingBuffer& buffer)
{
	Mat img;
	CascadeClassifier cascade;
	vector<Rect> pats;
	if(!cascade.load("/home/pi/seniordesign/classifiers/back/cascade.xml")){
		running = false;
		cout << "Error loading cascade" << endl;
		return;
	}
	unique_lock<mutex> lk(mtx);
	bufCV.wait(lk, []{return buffersReady;});
	lk.unlock();
	
	for(int i = 0; i < 10; i++){
		img = buffer.getLast();
		equalizeHist(img, img);
		
		cascade.detectMultiScale(img, pats, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
		for (size_t i = 0; i < pats.size(); i++)
		{
			Point center(pats[i].x + pats[i].width/2, pats[i].y + pats[i].height/2);
			ellipse(img, center, Size( pats[i].width/2, pats[i].height/2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		}
		imwrite(format("img%d.bmp",i), buffer.getLast());
	}
	
	running = false;
}

int main()
{
	RaspiCam_Cv camera;
	RingBuffer buffer;
	camera.set(CV_CAP_PROP_FORMAT, CV_8UC1);
	camera.open();
	thread cap(grabAndRetrieveImage, ref(camera), ref(buffer));
	thread disp(display, ref(buffer));
	cap.join();
	disp.join();
	return 0;
}
