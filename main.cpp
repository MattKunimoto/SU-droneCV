//Version 0.1 Drone CV
//General framework of how OpenCV works in C, needs RaspiCam additions:
// https://www.uco.es/investiga/grupos/ava/node/40
//

#include </usr/local/include/opencv2/core/core.hpp>
#include </usr/local/include/opencv2/highgui/highgui.hpp>
#include </usr/local/include/opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <thread>
#include <mutex>

std::mutex bufferUse;

void cameraFeed(cv::Mat& buffer)
{
	//import camera feed and save image to sd card
	bufferUse.lock();
	buffer = cv::imread("pic.jpg");
	bufferUse.unlock();
}

void harrisCorners(cv::Mat& buffer)
{
	//swap image through pipe like object or locks and pass by reference
	cv::Mat img, gray, harris, harris_norm, harris_norm_scaled;
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
}

int main( void )
{
	// 1. initialize camera and shit
	// 2. call thread of cameraFeed
	// 3. call thread of harris corners

	cv::Mat buffer;

	std::thread camFeed(cameraFeed, std::ref(buffer));
	std::thread harris(harrisCorners, std::ref(buffer));

	camFeed.join();
	harris.join();

	return 0;
}
