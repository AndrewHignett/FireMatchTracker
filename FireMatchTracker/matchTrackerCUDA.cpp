#include <iostream>
#include <stdio.h>
#include <gl/glut.h>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
#include <opencv2/cudaarithm.hpp>
using namespace cv::cuda;
#include "matchTracker.h"

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

void display() {
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0, 0.0, 0.0);

	glBegin(GL_POINTS);
	glVertex2f(10.0, 10.0);
	glVertex2f(150.0, 80.0);
	glVertex2f(100.0, 20.0);
	glVertex2f(200.0, 100.0);
	glEnd();
	glFlush();
}

void myinit() {
	glClearColor(1.0, 1.0, 1.0, 1.0);
	glColor3f(1.0, 0.0, 0.0);
	glPointSize(5.0);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, 499.0, 0.0, 499.0);
}

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
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

	int bufferedFrameCount = 0;
	Mat *frameBuffer = new Mat[3];
	namedWindow("frame", 1);
	for (;;)
	{
		glutInit(&argc, argv);
		//glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
		//glutInitWindowSize(500, 500);
		//glutInitWindowPosition(0, 0);
		//glutCreateWindow("Points");
		//glutDisplayFunc(display);

		//myinit();
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