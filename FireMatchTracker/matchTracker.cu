#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "matchTracker.h"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include "opencv2/cudaimgproc.hpp"
using namespace cv::cuda;


#define X 1280
#define Y 720

//Not ideal, expectedly produces motion blur, but not in a nice way
__global__ void averageKernel(cv::cuda::GpuMat out, cv::cuda::PtrStepSz<uint8_t[3]> bufferFrames[3])
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < X * Y)
	{
		int row = threadId / X;
		int column = threadId % X;
		bool colour = false;
		for (int i = 0; i < 3; i++) {
			//printf("1. %d\n", &bufferFrames[i]);
			//cv::cuda::GpuMat thisMat = (cv::cuda::GpuMat) *bufferFrames[i];
			//uint8_t B = bufferFrames[i](row, column)[0];
			uint8_t G = bufferFrames[i](row, column)[1];
			//uint8_t R = bufferFrames[i](row, column)[2];
			//if ((B > 0)||(G > 0)||(R > 0)) {
			//	printf("%d %d %d\n", B, G, R);
			//}

			if (G > 0) {
				//printf("%d %d\n", row, column);
				colour = true;
			}
			
			//printf("%d %d %d\n", row, column, bufferFrames[i](row, column));
			//if (bufferFrames[i].ptr(row)[column] > 0) {
				//printf("%d %d %f\n", row, column, bufferFrames[i].ptr(row)[column]);
			//}			
			//if (bufferFrames[i].data[(row*bufferFrames[i].step) + column * 3 + 1] > 0) {
			//	colour = true;
			//	printf(":D\n");
			//}
		}
		

		if (colour)
		{
			out.data[(row*out.step) + column * 3] = 0;
			out.data[(row*out.step) + column * 3 + 1] = 0;
			out.data[(row*out.step) + column * 3 + 2] = 255;
		}
	}
}

__global__ void detectObjectKernel(uint8_t *a, cv::cuda::GpuMat cleanFrame, cv::cuda::GpuMat frameCopy)
{
	//detect object size here
	printf("test\n");
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < X * Y)
	{
		int row = threadId / X;
		int column = threadId % X;
		printf("%d", cleanFrame.data[(row*cleanFrame.step) + column * 3 + 1]);
		if (cleanFrame.data[(row*cleanFrame.step) + column * 3 + 1] > 0) {
			int maxX = column;
			int maxY = row;
			int minX = column;
			int minY = row;
			bool traversable = true;
			int pixelList[2][X*Y];
			pixelList[0][0] = column;
			pixelList[1][0] = row;
			int listLength = 1;
			frameCopy.data[row*frameCopy.step + column * 3 + 1] = 0;
			while (traversable){
				int newPixels = 0;
				for (int i = 0; i < listLength; i++) {
					int x = pixelList[0][i];
					int y = pixelList[1][i];
					if (x < X){
						if (frameCopy.data[y*frameCopy.step + (x + 1) * 3 + 1] > 0) {
							pixelList[0][listLength] = x + 1;
							pixelList[1][listLength] = y;
							listLength++;
							newPixels++;
							frameCopy.data[y*frameCopy.step + (x + 1) * 3 + 1] = 0;
							maxX++;
						}
					}
					if (x > 0){
						if (frameCopy.data[y*frameCopy.step + (x - 1) * 3 + 1] > 0) {
							pixelList[0][listLength] = x - 1;
							pixelList[1][listLength] = y;
							listLength++;
							newPixels++;
							frameCopy.data[y*frameCopy.step + (x - 1) * 3 + 1] = 0;
							minX--;
						}
					}
					if (y < Y){
						if (frameCopy.data[(y + 1)*frameCopy.step + x * 3 + 1] > 0) {
							pixelList[0][listLength] = x;
							pixelList[1][listLength] = y + 1;
							listLength++;
							newPixels++;
							frameCopy.data[(y + 1)*frameCopy.step + x * 3 + 1] = 0;
							maxY++;
						}
					}
					if (y > 0){
						if (frameCopy.data[(y - 1)*frameCopy.step + x * 3 + 1] > 0) {
							pixelList[0][listLength] = x;
							pixelList[1][listLength] = y - 1;
							listLength++;
							newPixels++;
							frameCopy.data[(y - 1)*frameCopy.step + x * 3 + 1] = 0;
							minY--;
						}
					}
				}
				if (newPixels == 0) {
					traversable = false;
					printf("%d %d %d %d\n", minX, maxX, minY, maxY);
				}
			}
		}
	}
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
			//for (int i = -5; i < 6; i++)
			//for (int i = -8; i < 9; i++)
			for (int i = -2; i < 3; i++)
			{
				//for (int j = -5; j < 6; j++)
				//for (int j = -8; j < 9; j++)
				for (int j = -2; j < 3; j++)
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
			//for (int i = -8; i < 9; i++)
			for (int i = -6; i < 7; i++)
			{
				for (int j = -6; j < 7; j++)
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
		if ((pixelR > 80) && (pixelB < 10) && (pixelG < 10))
		//if (((pixelR > 128) && (pixelB < 10) && (pixelG < 10))||((pixelR > 100)&&(pixelB < 4)&&(pixelG < 4)) || ((pixelR > 90) && (pixelB < 1) && (pixelG < 1)))
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

	uint8_t *trackingLocations = (uint8_t*)malloc(3 * 100 * sizeof(uint8_t));
	uint8_t *d_trackingLocations;
	uint8_t *d_copyFramePtr;
	cv::cuda::GpuMat d_copyFrame;
	d_erodedFrame.copyTo(d_copyFrame);
	//Allocate new device memory
	cudaMalloc((void**)&d_copyFramePtr, d_copyFrame.rows*d_copyFrame.step);
	cudaMemcpyAsync(d_copyFramePtr, d_copyFrame.ptr<uint8_t>(), d_copyFrame.rows*d_copyFrame.step, cudaMemcpyDeviceToDevice);

	cudaMalloc((void**)&d_trackingLocations, sizeof(uint8_t) * 3 * 100);
	detectObjectKernel<<<blocks, threadCount>>>(d_trackingLocations, d_erodedFrame, d_copyFrame);

	//preventing memory leaks, in the wrong positon right now, purposely
	free(trackingLocations);
	cudaFree(d_trackingLocations);
	cudaFree(d_copyFramePtr);


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

Mat averageFrame(Mat buffer[3]) {
	int threadCount = 1024;
	int blocks = (X * Y - 1) / threadCount + 1;
	if (blocks == 1)
	{
		threadCount = X * Y;
	}
	
	//copy buffer frames to device memory GpuMat
	//cv::cuda::GpuMat d_bufferFrames[3];
	//d_bufferFrames[0].upload(buffer[0]);
	//d_bufferFrames[1].upload(buffer[1]);
	//d_bufferFrames[2].upload(buffer[2]);
	
	
	//cv::cuda::PtrStepSz<float> *bufferPtr;
	//cv::cuda::PtrStepSz<float> d_arr[3];
	cv::cuda::PtrStepSz<uint8_t[3]> *bufferPtr;
	cv::cuda::PtrStepSz<uint8_t[3]> d_arr[3];
	cv::cuda::GpuMat d_bufferFrames[3];
	for (int i = 0; i < 3; i++) {
		d_bufferFrames[i].upload(buffer[i]);
		d_arr[i] = d_bufferFrames[i];
	}
	cudaMalloc((void**)&bufferPtr, sizeof(cv::cuda::PtrStepSz<uint8_t[3]>) * 3);
	cudaMemcpy(bufferPtr, d_arr, sizeof(cv::cuda::PtrStepSz<uint8_t[3]>) * 3, cudaMemcpyHostToDevice);
	
	//cudaMalloc((void**)&bufferPtr, sizeof(cv::cuda::PtrStepSz<float>)*3);
	//cudaMemcpy(bufferPtr, d_arr, sizeof(cv::cuda::PtrStepSz<float>) * 3, cudaMemcpyHostToDevice);

	uint8_t *d_bufferPtr;
	uint8_t *d_outPtr;
	cv::cuda::GpuMat d_outFrame;
	d_bufferFrames[0].copyTo(d_outFrame);

	cudaMalloc((void **)&d_outPtr, d_outFrame.rows*d_outFrame.step);
	cudaMemcpyAsync(d_outPtr, d_outFrame.ptr<uint8_t>(), d_outFrame.rows*d_outFrame.step, cudaMemcpyDeviceToDevice);

	//convert the frame to be completely black to avoid weird artifacts
	blackKernel<<<blocks, threadCount>>>(d_outFrame);

	//allocate new device memory
	cudaMalloc((void**)&d_bufferPtr, 3*d_bufferFrames[0].rows*d_bufferFrames[0].step);
	cudaMemcpyAsync(d_bufferPtr, d_bufferFrames[0].ptr<uint8_t>(), 3 * d_bufferFrames[0].rows*d_bufferFrames[0].step, cudaMemcpyDeviceToDevice);

	averageKernel<<<blocks, threadCount>>>(d_outFrame, bufferPtr);//d_bufferFrames);

	//free buffer pointer from device memory
	cudaFree(d_bufferPtr);
	d_bufferFrames[0].release();
	d_bufferFrames[1].release();
	d_bufferFrames[2].release();

	Mat outFrame;
	//d_outFrame.download(outFrame);
	//free out pointer from device memory
	cudaFree(d_outPtr);
	d_outFrame.release();


	//return outFrame;
	//for the sake of debugging
	return buffer[0];
}