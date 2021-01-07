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

#define WINDOW_TITLE "Window"
//const int MaxParticles = 10000;
//const int emissionsPerFrame = 100;
//should be variable based on distance
int matchWidth = 20;

//Untseted approach to a quicksort
bool operator<(Particle& x, Particle& y)
{
	return x.getLife() < y.getLife();
}

void updateBuffer(Mat buffer[3], Mat newFrame, int currentSize) {
	if (currentSize >= 3) {
		Mat temp1, temp2;
		buffer[2].copyTo(temp1);
		newFrame.copyTo(buffer[2]);
		buffer[1].copyTo(temp2);
		temp1.copyTo(buffer[1]);
		temp2.copyTo(buffer[0]);
	}
	else {
		printf("%d\n", currentSize);
		buffer[currentSize] = newFrame;
	}

}

int main(int argc, char** argv) {
	//test window setup
	//glfwInit();
	//GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, NULL, NULL);
	//glfwMakeContextCurrent(window);
	//glewExperimental = GL_TRUE;

	//if (glewInit() != GLEW_OK) {
		//error with initialisation
		//glfwTerminate();
	//}

	//try to have a particle effect on a transparent background and then apply to each frame

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
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

	int bufferedFrameCount = 0;
	Mat *frameBuffer = new Mat[3];
	//int *MaxParticles;
	//MaxParticles = (int*)malloc(sizeof(int));
	//*MaxParticles = 10000;
	Particle *particleContainer = (Particle*)malloc(sizeof(Particle) * MaxParticles);
	int *trackingLocation = (int*)malloc(sizeof(int) * 2);
	//access violation need to be malloced
	//glm::vec3 pos = { 0.0f, 0.0f, 0.0f };
	//glm::vec3 vel = { 0.0f, 0.0f, 0.0f };
	//float pos[3] = { 0.0, 0.0, 0.0 };
	//float vel[3] = { 0.0, 0.0, 0.0 };
	//unsigned char colour[4] = { 0, 0, 0, 0 };
	//float size = 1;
	//angle and weight may be unnessecary for this particle system
	//float angle = 0;
	//float weight = 1;
	//float life = 0;
	*particleContainer = *initialSetValues(particleContainer);
	namedWindow("frame", 1);
	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera
		//cv::cuda::GpuMat imageGpu;
		//imageGpu.upload(frame); //convert captured frame to gpu
		//cv::cuda::cvtColor(imageGpu, edges, CV_BGR2GRAY); 
		//edges.download(frame); //convert edges from gpu to host
		//GaussianBlur(frame, frame, Size(7, 7), 1.5, 1.5);
		//Canny(frame, frame, 0, 30, 3);
		//imshow("frame", frame);
		//Type is CV_U8 = unsigned char

		//keep initial frame image as well as tracking overlay
		//Mat outFrame = track(frame);
		track(frame, trackingLocation);
		//updateBuffer(frameBuffer, outFrame, bufferedFrameCount);
		//bufferedFrameCount++;
		//if (bufferedFrameCount > 2){
		//averageFrame(frameBuffer).copyTo(outFrame);
		//imshow("frame", outFrame);
		//Sort the particle list
		std::sort(particleContainer, particleContainer + MaxParticles);
		//merge_sort(ParticleContainer, 0, MaxParticles - 1);
		*particleContainer = *updateParticles(particleContainer, trackingLocation);
		Mat frameCopy(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
		//no need to sort before adding the flame as each particle can be tested in parrallel
		Mat flameFrame = addFlame(frameCopy, particleContainer);
		//imshow("frame", flameFrame);
		imshow("frame", frameCopy);
		if (waitKey(30) >= 0) break;
		waitKey(1);
		//}
	}
	destroyAllWindows();
	return 0;
}