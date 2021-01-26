#include <iostream>
#include <stdio.h>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
#include <opencv2/cudaarithm.hpp>
using namespace cv::cuda;
#include "matchTracker.h"
#include "Particle.h"

//operator change for implementing the merge sort
bool operator<(Particle& x, Particle& y)
{
	return x.getLife() < y.getLife();
}

/*
Main function for match tracking. Access the webcam and display the image in real time. Pass the frame into the
tracker, and find a tracking location, then emit particles from the tracking location and display the final
frame.
*/
int main(int argc, char** argv) {
	 //test code from Get started with OpenCV CUDA cpp
	printShortCudaDeviceInfo(getDevice());
	int cuda_devices_number = getCudaEnabledDeviceCount();
	cout << "CUDA Device(s) Number: " << cuda_devices_number << endl;
	DeviceInfo _deviceInfo;
	bool _isd_evice_compatible = _deviceInfo.isCompatible();
	cout << "CUDA Device(s) Compatible: " << _isd_evice_compatible << endl;

	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;
	cap.set(CV_CAP_PROP_FRAME_WIDTH, X);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, Y);

	//Allocate memory to the array of particles to make up the flame
	Particle *particleContainer = (Particle*)malloc(sizeof(Particle) * MaxParticles);
	int *trackingLocation = (int*)malloc(sizeof(int) * 2);
	*particleContainer = *initialSetValues(particleContainer);
	namedWindow("Match Tracker", 1);
	Mat frame;
	//infinite loop capturing frames
	for (;;)
	{
		cap >> frame; // get a new frame from camera		
		track(frame, trackingLocation);
		//Sort the particle list
		std::sort(particleContainer, particleContainer + MaxParticles);
		Mat flameFrame(Y, X, CV_8UC3, cv::Scalar(0, 0, 0));
		*particleContainer = *updateParticles(particleContainer, trackingLocation);
		//no need to sort before adding the flame as each particle can be tested in parrallel
		addFlame(flameFrame, frame, particleContainer);
		imshow("Match Tracker", flameFrame);
		if (waitKey(30) >= 0) break;
		waitKey(1);
	}
	destroyAllWindows();
	return 0;
}