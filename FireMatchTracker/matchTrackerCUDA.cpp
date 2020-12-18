#include <iostream>
#include <stdio.h>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
#include <opencv2/cudaarithm.hpp>
using namespace cv::cuda;
#include "matchTracker.h"

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720
#define WINDOW_TITLE "Window"
const int MaxParticles = 100000;
Particle ParticleContainer[MaxParticles];
int LastUsedParticle = 0;


//For the flame setup
//Particle definition
class Particle {
	//glm::vec4 Position;
	//glm::vec4 velocity;
	//glm::vec4 Color;
	glm::vec3 position, velocity, acceleration;
	unsigned char r, g, b, a; //colour and alpha
	float size, angle, weight;
	float life; //remaining life of the particle. Dead and unused if < 0
public:
	void setValues(glm::vec3, glm::vec3, glm::vec3, unsigned char[4], float, float, float, float);
	void updateParticle(float);
};

void Particle::setValues(glm::vec3 pos, glm::vec3 vel, glm::vec3 acc, unsigned char colour[4], float sizeI, float angleI, float weightI, float lifeI) {
	position = pos;
	velocity = vel;
	acceleration = acc;
	r = colour[0];
	g = colour[1];
	b = colour[2];
	a = colour[3];
	size = sizeI;
	angle = angleI;
	weight = weightI;
	life = lifeI;
}

//Update the state parameters of the particle based off it's acceleration, initial velocity and position
void Particle::updateParticle(float deltaT) {
	//update the values for this particle
	//postion =
	//velocity =
	//acceleration =
	//life = 
	//Later, update colour based roughly off life span to emulate smoke or edge colouring of the fire
}

//class container for the particle system, may be unnecessary
class ParticleSystem {

};

//code intended for opengl use
int findUnusedParticle() {
	for (int i = LastUsedParticle; i < MaxParticles; i++) {
		if (ParticleContainer[i].life < 0) {
			LastUsedParticle = i;
			return i;
		}
	}

	for (int i = 0; i < LastUsedParticle; i++) {
		if (ParticleContainer[i].life < 0) {
			LastUsedParticle = i;
			return i;
		}
	}

	return 0; //all particles are taken
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
		Mat outFrame = track(frame);
		//updateBuffer(frameBuffer, outFrame, bufferedFrameCount);
		//bufferedFrameCount++;
		//if (bufferedFrameCount > 2){
		//averageFrame(frameBuffer).copyTo(outFrame);
		imshow("frame", outFrame);
		if (waitKey(30) >= 0) break;
		waitKey(1);
		//}
	}
	destroyAllWindows();
	return 0;
}