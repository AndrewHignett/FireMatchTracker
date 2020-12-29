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

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720
#define WINDOW_TITLE "Window"
const int MaxParticles = 10000;
const float FrameTime = 0.033;
const int emissionsPerFrame = 100;
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
	Particle *ParticleContainer[MaxParticles];
	//access violation need to be malloced
	glm::vec3 pos = { 0.0f, 0.0f, 0.0f };
	glm::vec3 vel = { 0.0f, 0.0f, 0.0f };
	unsigned char colour[4] = { 0, 0, 0, 0 };
	float size = 1;
	//angle and weight may be unnessecary for this particle system
	float angle = 0;
	float weight = 1;
	float life = 0;
	ParticleContainer[0]->setValues(pos, vel, colour, size, angle, weight, life);
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
		int* trackingLocation = track(frame);
		//updateBuffer(frameBuffer, outFrame, bufferedFrameCount);
		//bufferedFrameCount++;
		//if (bufferedFrameCount > 2){
		//averageFrame(frameBuffer).copyTo(outFrame);
		//imshow("frame", outFrame);
		//Sort the particle list
		std::sort(ParticleContainer, ParticleContainer + MaxParticles);
		//merge_sort(ParticleContainer, 0, MaxParticles - 1);
		Particle *ParticleContainer = updateParticles(FrameTime, ParticleContainer, MaxParticles, emissionsPerFrame);
		Mat flameFrame = addFlame(frame, trackingLocation, ParticleContainer, MaxParticles);
		imshow("frame", flameFrame);
		if (waitKey(30) >= 0) break;
		waitKey(1);
		//}
	}
	destroyAllWindows();
	return 0;
}