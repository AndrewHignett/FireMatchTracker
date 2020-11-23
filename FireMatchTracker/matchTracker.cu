#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "matchTracker.h"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include "opencv2/cudaimgproc.hpp"
#include <set>
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

//store the length of trackingLocations in shared memory, then sync threads and store it in trackingLocations[0];
__global__ void detectObjectKernel(std::set<int[2]> trackingLocations, cv::cuda::GpuMat trackedFrame, cv::cuda::GpuMat cleanFrame, cv::cuda::GpuMat frameCopy)//, int *a , cv::cuda::GpuMat cleanFrame, cv::cuda::GpuMat frameCopy)int *trackingLocations, 
{
	//detect object size here
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int row = threadId / X;
	int column = threadId % X;
	if (threadId < X * Y) {	
		trackedFrame.data[(row*trackedFrame.step) + column * 3] = 0;
		trackedFrame.data[(row*trackedFrame.step) + column * 3 + 1] = 0;
		trackedFrame.data[(row*trackedFrame.step) + column * 3 + 2] = 0;
	}
	if ((threadId < X * Y) && ((threadId/X)%10 == 0) && ((threadId % X) % 10 == 0))
	{
		uint8_t pixelGClean = cleanFrame.data[(row*cleanFrame.step) + column * 3 + 1];
		if (pixelGClean == 255) {
			int maxX = column;
			int maxY = row;
			int minX = column;
			int minY = row;
			bool traversable = true;
			//Current array size is unstable, only works up until half the pixels being red (displayed as green), but saves memory and it's expected there's effort 
			//to be as few red pixels as possible
			int pixelList[2][(X * Y) / 200];
			bool pixelUsed[X / 10][Y / 10] = { 0 };
			pixelList[0][0] = column;
			pixelList[1][0] = row;
			int listLength = 1;
			frameCopy.data[row*frameCopy.step + column * 3 + 1] = 0;
			while (listLength > 0) {
				int x = pixelList[0][0];
				int y = pixelList[1][0];
				if (!pixelUsed[x / 10][y / 10]) {
					if (x < X - 10) {
						uint8_t pixelGcleanTest = cleanFrame.data[y*cleanFrame.step + (x + 10) * 3 + 1];
						if ((pixelGcleanTest == 255) && (!pixelUsed[(x + 10) / 10][y / 10])) {
							pixelList[0][listLength] = x + 10;
							pixelList[1][listLength] = y;
							listLength++;
							if (x + 10 > maxX) {
								maxX = maxX + 10;
							}
						}
					}
					if (x > 9) {
						uint8_t pixelGcleanTest = cleanFrame.data[y*cleanFrame.step + (x - 10) * 3 + 1];
						if ((pixelGcleanTest == 255) && (!pixelUsed[(x - 10) / 10][y / 10])) {
							pixelList[0][listLength] = x - 10;
							pixelList[1][listLength] = y;
							listLength++;
							if (x - 10 < minX) {
								minX = minX - 10;
							}
						}
					}
					if (y < Y - 10) {
						uint8_t pixelGcleanTest = cleanFrame.data[(y + 10)*cleanFrame.step + x * 3 + 1];
						if ((pixelGcleanTest == 255) && (!pixelUsed[x / 10][(y + 10) / 10])) {
							pixelList[0][listLength] = x;
							pixelList[1][listLength] = y + 10;
							listLength++;
							if (y + 10 > maxY) {
								maxY = maxY + 10;
							}
						}
					}
					if (y > 9) {
						uint8_t pixelGcleanTest = cleanFrame.data[(y - 10)*cleanFrame.step + x * 3 + 1];
						if ((pixelGcleanTest == 255) && (!pixelUsed[x / 10][(y - 10) / 10])) {
							pixelList[0][listLength] = x;
							pixelList[1][listLength] = y - 10;
							listLength++;
							if (y - 10 < minY) {
								minY = minY - 10;
							}
						}
					}
				}
				pixelUsed[x / 10][y / 10] = true;
				int xTemp = pixelList[0][listLength - 1];
				int yTemp = pixelList[1][listLength - 1];
				pixelList[0][listLength - 1] = pixelList[0][0];
				pixelList[1][listLength - 1] = pixelList[1][0];
				pixelList[0][0] = xTemp;
				pixelList[1][0] = yTemp;
				listLength--;
			}
			int centreX = (maxX - minX) / 2 + minX;
			int centreY = (maxY - minY) / 2 + minY;
			trackedFrame.data[(centreY*trackedFrame.step) + centreX * 3 + 2] = 255;
			//trackedFrame.data[(maxY*trackedFrame.step) + maxX * 3 + 2] = 255;
			//trackedFrame.data[(maxY*trackedFrame.step) + minX * 3 + 2] = 255;
			//trackedFrame.data[(minY*trackedFrame.step) + maxX * 3 + 2] = 255;
			//trackedFrame.data[(minY*trackedFrame.step) + minX * 3 + 2] = 255;
			//trackingLocations[1 + threadId/100] = centreX;
			trackingLocations.insert({ centreX, centreY });
			//trackingLocations[2 + threadId/100] = centreY;

		}
	}
	__syncthreads();
	//list length includes repeated tracking locations
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
		//if ((pixelR > 128) && (pixelB < 50) && (pixelG < 50))
		if ((pixelR > 80) && (pixelB < 10) && (pixelG < 10))
		//if (((pixelR > 128) && (pixelB < 10) && (pixelG < 10))||((pixelR > 100)&&(pixelB < 4)&&(pixelG < 4)) || ((pixelR > 90) && (pixelB < 1) && (pixelG < 1)))
		//if (((pixelR > 100) && (pixelB < 5) && (pixelG < 5)))
		//if ((pixelR > 4*(pixelB + pixelG))&&(pixelR > 110))
		{
			out.data[(row*out.step) + column * 3] = 0;
			out.data[(row*out.step) + column * 3 + 1] = 0;
			out.data[(row*out.step) + column * 3 + 2] = 255;
		}
		else
		{
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

int* getMatchLocation(std::set<int[2]> trackingLocations){
	//for i {
		//for j {
			//if i != j
			//for k {
				//if i != k and j != k
				//check dot product of vectors between i and j (a), and j and k (b)
				//divided by the product of magnitudes of |a| and |b|, this should be close to 1
				//work out magnitude of a and b, b should be approximately double a
				//The one that is closest to 1 with b approximately double a is the correct match position
			//}
		//}
	//}
	double finalDotProduct = 0;	
	double finalRatio;
	//iterate over set to find all 3 location combinations, and find the most likely one to be the matchstick
	for (auto i : trackingLocations) {
		for (auto j : trackingLocations) {
			if ((i[0] != j[0]) || (i[1] != j[1])) {
				for (auto k : trackingLocations) {
					if (((i[0] != k[0]) || (i[1] != k[1])) && ((j[0] != k[0]) || (j[1] != k[1]))) {
						int a[2] = { j[0] - i[0], j[1] - i[1] };
						int b[2] = { k[0] - j[0], k[1] - j[1] };
						double aMagnitude = sqrt(a[0] * a[0] + a[1] * a[1]);
						double bMagnitude = sqrt(b[0] * b[0] + b[1] * b[1]);
						double dotProduct = sqrt(a[0] * b[0] + a[1] * b[1])/(aMagnitude * bMagnitude);
						double ratio = bMagnitude/aMagnitude;
						if (abs(1 - finalDotProduct) > abs(1 - dotProduct)) {
							//test if ratio is close to 2, this does not have to be the closest to 2
							//but instead should just be close enough, perhaps within a valid range
						}
					}
				}
			}			
		}
	}
}

Mat track(Mat frame) {
	int threadCount = 1024;
	int blocks = (X * Y - 1) / threadCount + 1;
	if (blocks == 1)
	{
		threadCount = X * Y;
	}
	//Set up device variables
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
	getRedKernel << <blocks, threadCount >> > (d_outFrame, d_newFrame);
	cudaDeviceSynchronize();

	//Free original frame pointer device memory
	cudaFree(d_imgPtr);
	d_newFrame.release();

	uint8_t *d_dilatedPtr;
	cv::cuda::GpuMat d_dilatedFrame;
	d_outFrame.copyTo(d_dilatedFrame);

	//Allocate new device memory
	cudaMalloc((void**)&d_dilatedPtr, d_dilatedFrame.rows*d_dilatedFrame.step);
	cudaMemcpyAsync(d_dilatedPtr, d_dilatedFrame.ptr<uint8_t>(), d_dilatedFrame.rows*d_dilatedFrame.step, cudaMemcpyDeviceToDevice);

	dilateKernel << <blocks, threadCount >> > (d_dilatedFrame, d_outFrame);
	cudaDeviceSynchronize();
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
	blackKernel << <blocks, threadCount >> > (d_erodedFrame);
	cudaDeviceSynchronize();
	erodeKernel << <blocks, threadCount >> > (d_erodedFrame, d_dilatedFrame);
	cudaDeviceSynchronize();
	//Free dilatedFrame pointer device memory
	cudaFree(d_dilatedPtr);
	d_dilatedFrame.release();

	//int *trackingLocations = (int*)malloc((X/10)*(Y/10) * sizeof(int));
	//int *d_trackingLocations;
	//std::set<int[2], greater<int[2]>> *trackingLocations = (std::set<int[2]>)malloc(sizeof(std::set<int[2]>));
	std::set<int[2], std::greater<int[2]>> *trackingLocations = (std::set<int[2], std::greater<int[2]>>*)malloc(2 * (X / 20)*(Y / 20) * sizeof(int));// = (std::set<int[2]>)malloc(sizeof(std::set<int[2]>));
	std::set<int[2], std::greater<int[2]>> *d_trackingLocations;
	uint8_t *d_copyFramePtr;
	cv::cuda::GpuMat d_copyFrame;
	uint8_t *d_trackedFramePtr;
	cv::cuda::GpuMat d_trackedFrame;
	d_erodedFrame.copyTo(d_copyFrame);
	d_erodedFrame.copyTo(d_trackedFrame);
	//Allocate new device memory
	cudaMalloc((void**)&d_copyFramePtr, d_copyFrame.rows*d_copyFrame.step);
	cudaMalloc((void**)&d_trackedFramePtr, d_trackedFrame.rows*d_trackedFrame.step);
	cudaMemcpyAsync(d_copyFramePtr, d_copyFrame.ptr<uint8_t>(), d_copyFrame.rows*d_copyFrame.step, cudaMemcpyDeviceToDevice);
	cudaMemcpyAsync(d_trackedFramePtr, d_trackedFrame.ptr<uint8_t>(), d_trackedFrame.rows*d_trackedFrame.step, cudaMemcpyDeviceToDevice);

	//cudaMalloc((void**)&d_trackingLocations, (X/10) * (Y/10) * sizeof(int));
	cudaMalloc((void**)&d_trackingLocations, 2 * (X / 20) * (Y / 20) * sizeof(int));
	
	detectObjectKernel<<<blocks, threadCount>>>(d_trackingLocations, d_trackedFrame, d_erodedFrame, d_copyFrame);
	cudaError_t error2 = cudaGetLastError();
	if (error2 != cudaSuccess) {
		printf("2. Error: %s\n", cudaGetErrorString(error2));
	}
	cudaDeviceSynchronize();

	cudaMemcpy(trackingLocations, d_trackingLocations, 2 * (X/20) * (Y/20) * sizeof(int), cudaMemcpyDeviceToHost);
	//printf("%u %u %u %u\n", trackingLocations[0], trackingLocations[1], trackingLocations[2], trackingLocations[3]);
	//printf("%d %d %d %d\n", trackingLocations[0], trackingLocations[1], trackingLocations[2], trackingLocations[3]);
	print(trackingLocations);

	//preventing memory leaks, in the wrong positon right now, purposely
	free(trackingLocations);
	cudaFree(d_trackingLocations);
	cudaFree(d_copyFramePtr);
	d_copyFrame.release();

	Mat outFrame;
	d_trackedFrame.download(outFrame);
	//d_erodedFrame.download(outFrame);

	//Free erodedFrame pointer device memory
	cudaFree(d_erodedPtr);
	d_erodedFrame.release();

	//Free the tracked frame from device memory
	cudaFree(d_trackedFramePtr);
	d_trackedFrame.release();
	
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