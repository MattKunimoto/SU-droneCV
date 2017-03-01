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

mutex mtx1;
mutex mtx2;
mutex mtx3;
condition_variable bufCV;
bool buffersReady = false;
bool running = true;

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
	
	//overloaded assignement operator
	void operator=(const Mat& image){
		currentNode->img = image;
		currentNode = currentNode->next;
	}
	
	Mat getCurrent(){
		return currentNode->img;
	}
	
private:
	class Node{
	public:
		Node(){
			img = zeros(1280, 960, CV_U8C1);
			next = nullptr;
			prev = nullptr;
		}
		Mat img;
		Node* next = nullptr;
		Node* prev = nullptr;
	}
	
	Node * currentNode = nullptr;
}

void grabAndRetrieveImage(RaspiCam_Cv& camera, Mat& buffer1, Mat& buffer2, Mat& buffer3)
{
	unique_lock<mutex> lk1(mtx1);
	unique_lock<mutex> lk2(mtx2);
	unique_lock<mutex> lk3(mtx3);

	camera.grab();
	camera.retrieve(buffer1);
	lk1.unlock();
	
	cout << "1" << endl;
	
	camera.grab();
	camera.retrieve(buffer2);
	lk2.unlock();
	cout << "2" << endl;
	
	camera.grab();
	camera.retrieve(buffer3);
	cout << "3" << endl;
	buffersReady = true;
	lk3.unlock();
	bufCV.notify_one();
	
	while(running){
		if(lk1.try_lock()){
			camera.grab();
			camera.retrieve(buffer1);
			lk1.unlock();
		}
		
		if(lk2.try_lock()){
			camera.grab();
			camera.retrieve(buffer2);
			lk2.unlock();
		}
		
		if(lk3.try_lock()){
			camera.grab();
			camera.retrieve(buffer3);
			lk3.unlock();
		}
	}
}


void display(Mat& buffer1, Mat& buffer2, Mat& buffer3)
{
	Mat img;
	unique_lock<mutex> lk1(mtx1, defer_lock);
	unique_lock<mutex> lk2(mtx2, defer_lock);
	unique_lock<mutex> lk3(mtx3);
	bufCV.wait(lk3, []{return buffersReady;});
	lk3.unlock();
	
	if(!lk1.try_lock()){
		lk3.lock();
		img = buffer3;
		lk3.unlock();
		cout << "13" << endl;
	}else if(!lk2.try_lock()){
		lk1.lock();
		img = buffer1;
		lk1.unlock();
		cout << "11" << endl;
	}else if(!lk3.try_lock()){
		lk2.lock();
		img = buffer2;
		lk2.unlock();
		cout << "12" << endl;
	}
	
	imwrite("img1.bmp", img);
	
	if(!lk1.try_lock()){
		lk3.lock();
		img = buffer3;
		lk3.unlock();
		cout << "23" << endl;
	}else if(!lk2.try_lock()){
		img = buffer1;
		lk1.unlock();
		cout << "21" << endl;
	}else{
		lk1.unlock();
		img = buffer2;
		lk2.unlock();
		cout << "22" << endl;
	}
	
	imwrite("img2.bmp", img);
	
	if(!lk1.try_lock()){
		lk3.lock();
		img = buffer3;
		lk3.unlock();
		cout << "33" << endl;
	}else if(!lk2.try_lock()){
		img = buffer1;
		lk1.unlock();
		cout << "31" << endl;
	}else{
		lk1.unlock();
		img = buffer2;
		lk2.unlock();
		cout << "32" << endl;
	}
	
	imwrite("img3.bmp", img);
	
	if(!lk1.try_lock()){
		lk3.lock();
		img = buffer3;
		lk3.unlock();
		cout << "43" << endl;
	}else if(!lk2.try_lock()){
		img = buffer1;
		lk1.unlock();
		cout << "41" << endl;
	}else{
		lk1.unlock();
		img = buffer2;
		lk2.unlock();
		cout << "42" << endl;
	}
	
	imwrite("img4.bmp", img);
	
	if(!lk1.try_lock()){
		lk3.lock();
		img = buffer3;
		lk3.unlock();
		cout << "53" << endl;
	}else if(!lk2.try_lock()){
		img = buffer1;
		lk1.unlock();
		cout << "51" << endl;
	}else{
		lk1.unlock();
		img = buffer2;
		lk2.unlock();
		cout << "52" << endl;
	}
	
	imwrite("img5.bmp", img);
	
	if(!lk1.try_lock()){
		lk3.lock();
		img = buffer3;
		lk3.unlock();
		cout << "63" << endl;
	}else if(!lk2.try_lock()){
		img = buffer1;
		lk1.unlock();
		cout << "61" << endl;
	}else{
		lk1.unlock();
		img = buffer2;
		lk2.unlock();
		cout << "62" << endl;
	}
	
	imwrite("img6.bmp", img);
	
	
	running = false;
}

int main()
{
	RaspiCam_Cv camera;
	Mat buffer1;
	Mat buffer2;
	Mat buffer3;
	camera.set(CV_CAP_PROP_FORMAT, CV_8UC1);
	camera.open();
	thread cap(grabAndRetrieveImage, ref(camera), ref(buffer1), ref(buffer2), ref(buffer3));
	thread disp(display, ref(buffer1), ref(buffer2), ref(buffer3));
	cap.join();
	disp.join();
	return 0;
}
