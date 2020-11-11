#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "matchTracker.h"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include "opencv2/cudaimgproc.hpp"
using namespace cv::cuda;


#define X 1280
#define Y 720

__global__ void detectObjectKernel(cv::cuda::GpuMat out, cv::cuda::GpuMat cleanFrame)
{
	//detect object size here
}

__global__ void erodeKernel(cv::cuda::GpuMat out, cv::cuda::GpuMat dilatedFrame)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < X * Y)
	{
		int row = threadId / X;
		int column = threadId % X;
		uint8_t pixelR = dilatedFrame.data[(row*dilatedFrame.step) + column * 3 + 2];

		if (pixelR == 255)
		{
			bool allPixelsRed = true;
			for (int i = -8; i < 9; i++)
			{
				for (int j = -8; j < 9; j++)
				{
					if ((row + i > -1) && (row + i < Y) && (column + j > -1) && (column + j < X))
					{
						if (dilatedFrame.data[((row + i)*dilatedFrame.step) + (column + j) * 3 + 2] == 0)
						{
							allPixelsRed = false;
						}
					}
				}
			}
			if (allPixelsRed)
			{
				out.data[(row*out.step) + column * 3] = 0;
				out.data[(row*out.step) + column * 3 + 1] = 255;
				out.data[(row*out.step) + column * 3 + 2] = 0;
			}
		}
	}
}

__global__ void dilateKernel(cv::cuda::GpuMat out, cv::cuda::GpuMat redFrame)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < X * Y)
	{
		int row = threadId / X;
		int column = threadId % X;
		uint8_t pixelR = redFrame.data[(row*redFrame.step) + column * 3 + 2];

		if (pixelR == 255)
		{
			for (int i = -8; i < 9; i++)
			{
				for (int j = -8; j < 9; j++)
				{
					if (!(i == 0 && j == 0))
					{
						if ((row + i > -1) && (row + i < Y) && (column + j > -1) && (column + j < X))
						{
							out.data[((row + i)*out.step) + (column + j) * 3 + 2] = 255;
						}
					}
				}
			}
		}		
	}
}

__global__ void getRedKernel(cv::cuda::GpuMat out, cv::cuda::GpuMat frame)
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
		//if ((pixelR > 128) && (pixelB < 50) && (pixelG < 50))
		if (((pixelR > 128) && (pixelB < 10) && (pixelG < 10))||((pixelR > 100)&&(pixelB < 4)&&(pixelG < 4)) || ((pixelR > 90) && (pixelB < 1) && (pixelG < 1)))
		//if (((pixelR > 100) && (pixelB < 5) && (pixelG < 5)))
		//if ((pixelR > 4*(pixelB + pixelG))&&(pixelR > 110))
		{

			//out.data[(row*out.step) + column * 3] = pixelB;
			//out.data[(row*out.step) + column * 3 + 1] = pixelG;
			//out.data[(row*out.step) + column * 3 + 2] = pixelR;
			out.data[(row*out.step) + column * 3] = 0;
			out.data[(row*out.step) + column * 3 + 1] = 0;
			out.data[(row*out.step) + column * 3 + 2] = 255;
		}
		else
		{
			//out.data[(row*out.step) + column * 3] = pixelB/2;
			//out.data[(row*out.step) + column * 3 + 1] = pixelG/2;
			//out.data[(row*out.step) + column * 3 + 2] = pixelR/2;
			out.data[(row*out.step) + column * 3] = 0;
			out.data[(row*out.step) + column * 3 + 1] = 0;
			out.data[(row*out.step) + column * 3 + 2] = 0;
		}
	}
}

__global__ void blackKernel(cv::cuda::GpuMat out)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < X * Y)
	{
		int row = threadId / X;
		int column = threadId % X;
		out.data[(row*out.step) + column * 3] = 0;
		out.data[(row*out.step) + column * 3 + 1] = 0;
		out.data[(row*out.step) + column * 3 + 2] = 0;
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
	getRedKernel<<<blocks, threadCount>>>(d_outFrame, d_newFrame);
	//Free newFrame device and host memory
	//cudaFree(d_newFrame);
	//free(newFrame);
	//cudaMemcpy(outFrame, d_outFrame, sizeof(frame), cudaMemcpyDeviceToHost);
	//Image dilation
	//int erosionDilation_size = 5;
	//Mat element = cv::getStructuringElement(MORPH_RECT, Size(2 * erosionDilation_size + 1, 2 * erosionDilation_size + 1));
	//Ptr<cuda::Filter> dilateFilter = cv::cuda::createMorphologyFilter(MORPH_DILATE, d_outFrame.type(), element);
	//dilateFilter->apply(d_outFrame, d_outFrame);
	
	//Free original frame pointer device memory
	cudaFree(d_imgPtr);
	d_newFrame.release();
	

	uint8_t *d_dilatedPtr;
	cv::cuda::GpuMat d_dilatedFrame;
	d_outFrame.copyTo(d_dilatedFrame);
	

	
	//Allocate new device memory
	cudaMalloc((void**)&d_dilatedPtr, d_dilatedFrame.rows*d_dilatedFrame.step);
	cudaMemcpyAsync(d_dilatedPtr, d_dilatedFrame.ptr<uint8_t>(), d_dilatedFrame.rows*d_dilatedFrame.step, cudaMemcpyDeviceToDevice);

	dilateKernel<<<blocks, threadCount>>>(d_dilatedFrame, d_outFrame);	
	
	//Free outFrame pointer device memory
	cudaFree(d_outPtr);
	d_outFrame.release();

	uint8_t *d_erodedPtr;
	cv::cuda::GpuMat d_erodedFrame;
	d_dilatedFrame.copyTo(d_erodedFrame);

	//Allocated new device memory
	cudaMalloc((void**)&d_erodedPtr, d_erodedFrame.rows*d_erodedFrame.step);
	cudaMemcpyAsync(d_erodedPtr, d_erodedFrame.ptr<uint8_t>(), d_erodedFrame.rows*d_erodedFrame.step, cudaMemcpyDeviceToDevice);

	//convert the frame to be completely black to avoid weird artifacts
	blackKernel<<<blocks, threadCount>>>(d_erodedFrame);
	
	erodeKernel<<<blocks, threadCount>>>(d_erodedFrame, d_dilatedFrame);
	
	//Free dilatedFrame pointer device memory
	cudaFree(d_dilatedPtr);
	d_dilatedFrame.release();

	Mat outFrame;
	d_erodedFrame.download(outFrame);
	
	//Free dilatedFrame pointer device memory
	cudaFree(d_erodedPtr);
	d_erodedFrame.release();
	
	return outFrame;
	//return *outFrame;
	//For the sake of debugging 
	return frame;
}