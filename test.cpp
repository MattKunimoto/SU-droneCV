//Current working version
//Runs at 4 fps at 640, 480
//Uses ~65% of RPi processing power

//TO DO:
//	Test initializing and joining classifier threads as opposed to pooling them
//	Take info in detection vector to do flight commands
//	Clean detection vector after interpreter for instructions acquires needed data

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

//buffer and classifier sync primitives
mutex bufMtx;
mutex classMtx;

condition_variable bufCV;
condition_variable classCV;

atomic_uchar classifiersFinished;

bool buffersReady = false;
bool startClassifiers = false;
bool running = false;
bool classificationDone = false;

static const char BACK = 'b';
static const char FRONT = 'f';
static const char SIDE = 's';
static const char NUM_PATTERNS = 3;
static const int FRAMES = 10000;

static const string backCascadeName = "/home/pi/seniordesign/classifiers/back/cascade.xml";
static const string frontCascadeName = "/home/pi/seniordesign/classifiers/front/cascade.xml";
static const string sideCascadeName = "/home/pi/seniordesign/classifiers/side/cascade.xml";

static const int WIDTH = 640;
static const int HEIGHT = 480;

// Creates triple buffer to store camera images
// Allows for classifier manager to grab newest image without contesting
//	for resources with camera feed
// Added 4th buffer due to contestion for some resource (unclear if this
//	buffer was being contested but issue hasn't arrised since switch)
class RingBuffer{
public:	
	// Constructor for ring buffer, creates 4 buffers in bi-directional circular link
	RingBuffer(){
		currentNode = new Node; //create buff1
		currentNode->next = new Node; //create buff2 as buff1 next
		currentNode->next->prev = currentNode; //link buff2 prev to buff1
		currentNode->next->next = new Node; //create buff3 as next for 2
		currentNode->next->next->prev = currentNode->next; //link 3 prev to 2
		currentNode->prev = new Node; //create buff4 as buff1 prev
		currentNode->prev->next = currentNode; //link buff4 next to buff1
		currentNode->prev->prev = currentNode->next->next; //link buff4 prev to buff3
		currentNode->next->next->next = currentNode->prev; //link buff3 next to buff4
	}
	
	// overloaded assignement operator to add images from camera feed easily
	void operator=(const Mat& image){
		currentNode->img = image;
		currentNode = currentNode->next;
	}
	
	// destructor to clean up buffers
	~RingBuffer(){
		delete currentNode->next->next;
		delete currentNode->next;
		delete currentNode->prev;
		delete currentNode;
	}
	
	//Access last saved image
	Mat getLast(){
		return currentNode->prev->img;
	}
	
private:
	//Nodes for buffer (each one is a buffer with links to the other 2)
	struct Node{
	public:
		Node(){
			img = Mat::zeros(WIDTH, HEIGHT, CV_8UC1);
			next = nullptr;
			prev = nullptr;
		}
		
		Mat img;
		Node* next;
		Node* prev;
	};
	
	//base node
	Node * currentNode = nullptr;
};

struct LocSizeSide
{
public:
	//base constructor does nothing
	LocSizeSide()
	{}
	
	//constructor to allow for 
	LocSizeSide(Rect rect, char s)
	{
		x = rect.x + (rect.width / 2); //x and y are set to middle
		y = rect.y + (rect.height / 2);
		w = rect.width;
		h = rect.height;
		side = s;
	}
	
	int x, y, w, h; //x and y are middle of pattern, not top left
	char side;
};

void cameraFeed(RaspiCam_Cv& camera, RingBuffer& buffer)
{
	unique_lock<mutex> lk(bufMtx);
	
	Mat img;
	
	//fill buffer
	for(int i = 0; i < 4; i++){
		camera.grab();
		camera.retrieve(img);
		buffer = img;
	}
	
	//buffer full, classifiers can run
	buffersReady = true;
	lk.unlock();
	bufCV.notify_one();
	
	//start image capture process
	while(running){
		camera.grab();
		camera.retrieve(img);
		buffer = img;
	}
}

void classifier(Mat& img, vector<LocSizeSide>& detections, CascadeClassifier cascade, char side)
{
	// Create local detection vector
	vector<Rect> localDetections;
	// Create lock and defer
	unique_lock<mutex> lk(classMtx, defer_lock);
	
	while(running){
		// Wait for classifier manager
		lk.lock();
		classCV.wait(lk, []{return startClassifiers;});
		
		// Allow other threads to unlock 
		// Matt is there a better way to implement a barrier in C++11?
		lk.unlock(); 
		classCV.notify_all();
		
		// Detect patterns
		cascade.detectMultiScale(img, localDetections, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
		
		// Add patterns to main detection vector including side info via emplacement which moves Rect data into LocSizeSide wrapper
		for(size_t i = 0; i < localDetections.size(); i++){
			detections.emplace_back(localDetections[i], side);
		}
		
		// Increment classifiers finished and notify manager, once at 3
		classifiersFinished++;
		startClassifiers = false;
		
		// If last classifier to finish allow classifier manager to prepare next frame
		if(classifiersFinished == NUM_PATTERNS)
			classificationDone = true;
		
		classCV.notify_all();
	}
	
}

void classifierManager(RingBuffer& buffer)
{
	Mat img;
	
	CascadeClassifier backCascade;
	//CascadeClassifier frontCascade; doesn't exist yet
	//CascadeClassifier sideCascade; doesn't exist yet
	
	vector<LocSizeSide> detections;
	
	if(!backCascade.load(backCascadeName)){
		running = false;
		cout << "Error loading back cascade" << endl;
		return;
	}
	
	/*if(!frontCascade.load(frontCascadeName)){
		running = false;
		cout << "Error loading front cascade" << endl;
		return;
	}
		THESE DON'T EXIST YET
	if(!sideCascade.load(sideCascadeName)){
		running = false;
		cout << "Error loading side cascade" << endl;
		return;
	}*/
	
	// Lock classifiers
	unique_lock<mutex> classLk(classMtx);
	
	// Currently just using back cascade on all 3
	thread backClassifier(classifier, ref(img), ref(detections), backCascade, BACK);
	thread frontClassifier(classifier, ref(img), ref(detections), backCascade, FRONT);
	thread sideClassifier(classifier, ref(img), ref(detections), backCascade, SIDE);
	
	// Wait for buffer to be filled
	unique_lock<mutex> bufLk(bufMtx);
	bufCV.wait(bufLk, []{return buffersReady;});

	cout << "starting classifiers" << endl;
	
	// Run time analysis
	time_point<system_clock> start, end;
	start = system_clock::now();
	
	// Currently for loop because there are no conditions to stop otherwise
	for(int i = 0; i < FRAMES; i++){
		// Grab image and prepare for classifiers
		img = buffer.getLast();
		equalizeHist(img, img);
		classifiersFinished = 0;
		classificationDone = false;
		
		// Tell classifiers to run
		startClassifiers = true;
		classLk.unlock();
		classCV.notify_all();
		
		// Wait until classifiers are finished running
		classLk.lock();
		classCV.wait(classLk, []{return classificationDone;});
		// ClassLk is reacquired after wait()
	}
	running = false;
	
	img = buffer.getLast();
	equalizeHist(img, img);
	classifiersFinished = 0;
	classificationDone = false;
	
	// Tell classifiers to run
	startClassifiers = true;
	classLk.unlock();
	classCV.notify_all();
	
	// Run time analysis
	end = system_clock::now();
	duration<double> elapsed_seconds = end-start;
	
	cout << "Exiting loop. " << FRAMES << " frames." << endl 
		<< "Total time: " << elapsed_seconds.count() << endl
		<< "Average frame rate: " << ((double) FRAMES / elapsed_seconds.count()) << endl;
	
	// End classifiers and camera feed
	backClassifier.join();
	frontClassifier.join();
	sideClassifier.join();
	
}

int main()
{
	// Create buffer
	RingBuffer buffer;
	
	// Create and set up camera
	RaspiCam_Cv camera;
	camera.set(CV_CAP_PROP_FORMAT, CV_8UC1); // Set camera to 8-bit grayscale
	camera.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH); // Set width
	camera.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT); // Set height
	camera.open();
	
	// Initialize threads
	running = true;
	thread cam(cameraFeed, ref(camera), ref(buffer));
	thread classMngr(classifierManager, ref(buffer));
	
	// Exit camera feed and classifier manager
	cam.join();
	classMngr.join();
	
	return 0;
}
