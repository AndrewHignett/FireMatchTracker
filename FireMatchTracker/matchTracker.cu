#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "matchTracker.h"
#include <opencv2/cudaarithm.hpp>
using namespace cv::cuda;

#define X 1280
#define Y 720

__global__ void trackKernel(cv::cuda::GpuMat out, cv::cuda::GpuMat frame)
{
	//detect end
	//determine orientation
	//determine distance
	//have internal representation of it's position in 3D
	//draw particles in 3D space
	//move particles with physics based on the match's movement
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < X * Y)
	{
		int row = threadId / X;
		int column = threadId % X;
		//BGR pixel values
		uint8_t pixelB = frame.data[(row*frame.step) + column * 3];
		uint8_t pixelG = frame.data[(row*frame.step) + column * 3 + 1];
		uint8_t pixelR = frame.data[(row*frame.step) + column * 3 + 2];
		//out.data[(row*out.step) + column * 3] = pixelB;
		//out.data[(row*out.step) + column * 3 + 1] = pixelG;
		//out.data[(row*out.step) + column * 3 + 2] = pixelR;
		if ((pixelR > 128) && (pixelB < 50) && (pixelG < 50))
		{
			out.data[(row*out.step) + column * 3] = pixelB;
			out.data[(row*out.step) + column * 3 + 1] = pixelG;
			out.data[(row*out.step) + column * 3 + 2] = pixelR;
		}
		else
		{
			out.data[(row*out.step) + column * 3] = 0;
			out.data[(row*out.step) + column * 3 + 1] = 0;
			out.data[(row*out.step) + column * 3 + 2] = 0;
		}
	}
}

Mat track(Mat frame) {
	//Mat *newFrame = (Mat*)malloc(X * Y * sizeof(Mat));
	//Mat *newFrame = (Mat*)malloc(sizeof(frame));
	//Mat *outFrame = (Mat*)malloc(sizeof(frame));
	//*newFrame = frame.clone();
	int threadCount = 1024;
	int blocks = (X * Y - 1) / threadCount + 1;
	if (blocks == 1)
	{
		threadCount = X * Y;
	}
	//Set up device variables
	//Mat *d_newFrame;
	//Mat *d_outFrame;
	uint8_t *d_imgPtr;
	uint8_t *d_outPtr;
	cv::cuda::GpuMat d_newFrame;
	cv::cuda::GpuMat d_outFrame;
	d_newFrame.upload(frame);
	d_outFrame.upload(frame);
	//Allocate device memory
	cudaMalloc((void **)&d_imgPtr, d_newFrame.rows*d_newFrame.step);
	cudaMalloc((void **)&d_outPtr, d_outFrame.rows*d_outFrame.step);
	cudaMemcpyAsync(d_imgPtr, d_newFrame.ptr<uint8_t>(), d_newFrame.rows*d_newFrame.step, cudaMemcpyDeviceToDevice);
	cudaMemcpyAsync(d_outPtr, d_outFrame.ptr<uint8_t>(), d_outFrame.rows*d_outFrame.step, cudaMemcpyDeviceToDevice);
	//cudaMalloc((void**)&d_newFrame, sizeof(frame));
	//cudaMalloc((void**)&d_outFrame, sizeof(frame));
	//transfer memory from host to device memory
	//cudaMemcpy(d_newFrame, newFrame, sizeof(frame), cudaMemcpyHostToDevice);
	trackKernel<<<blocks, threadCount>>>(d_outFrame, d_newFrame);
	//Free newFrame device and host memory
	//cudaFree(d_newFrame);
	//free(newFrame);
	//cudaMemcpy(outFrame, d_outFrame, sizeof(frame), cudaMemcpyDeviceToHost);
	//Free outFrame device memory
	cudaFree(d_imgPtr);
	cudaFree(d_outPtr);
	
	Mat outFrame;
	d_outFrame.download(outFrame);
	return outFrame;
	//return *outFrame;
	//For the sake of debugging 
	return frame;
}