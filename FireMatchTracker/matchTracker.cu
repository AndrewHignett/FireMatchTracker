#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace std;
#include <opencv2/core.hpp>
using namespace cv;
#include <opencv2/cudaarithm.hpp>
using namespace cv::cuda;

__global__ void trackKernel(int *c, const int *a, const int *b)
{
	//detect end
	//determine orientation
	//determine distance
	//have internal representation of it's position in 3D
	//draw particles in 3D space
	//move particles with physics based on the match's movement
}

int main()
{
	//test code from Get started with OpenCV CUDA cpp
	printShortCudaDeviceInfo(getDevice());
	int cuda_devices_number = getCudaEnabledDeviceCount();
	cout << "CUDA Device(s) Number: " << cuda_devices_number << endl;
	DeviceInfo _deviceInfo;
	bool _isd_evice_compatible = _deviceInfo.isCompatible();
	cout << "CUDA Device(s) Compatible: " << _isd_evice_compatible << endl;

	Mat frame;
	VideoCapture cam;// = VideoCapture(0, CAP_DSHOW);
	cam.open(0);// CAP_DSHOW);
	if (!cam.isOpened()) {
		cerr << "ERROR Unable to open camera\n";
		return -1;
	}
	//cam.read(frame);
	//if (frame.empty()) {
	//	cerr << "ERROR Blank frame\n";
	//	return -1;
	//}
	destroyAllWindows();


	//working code that opens and displays the webcam
	VideoCapture cap(0);
	while (1)
	{
		Mat image;
		cap >> image;
		if (!image.data) break;
		if (waitKey(30) >= 0) break;

		imshow("test", image);
		waitKey(1);
	}


	//Old basic CUDA code sample
	//--- INITIALIZE VIDEOCAPTURE
	//VideoCapture cap;
	// open the default camera using default API
	// cap.open(0);
	// OR advance usage: select any API backend
	//int deviceID = 0;             // 0 = open default camera
	//int apiID = cv::CAP_ANY;      // 0 = autodetect default API
	// open selected camera using selected API
	//cap.open(deviceID, apiID);
	// check if we succeeded
	/*if (!cap.isOpened()) {
		cerr << "ERROR! Unable to open camera\n";
		return -1;
	}
	
	//--- GRAB AND WRITE LOOP
	cout << "Start grabbing" << endl
		<< "Press any key to terminate" << endl;
	for (;;)
	{
		// wait for a new frame from camera and store it into 'frame'
		cap.read(frame);
		// check if we succeeded
		if (frame.empty()) {
			cerr << "ERROR! blank frame grabbed\n";
			break;
		}
		// show live and wait for a key with timeout long enough to show images
		imshow("Live", frame);
		if (waitKey(5) >= 0)
			break;
	}*/

	return 0;
}