#include "matchTracker.h"
#include <set>
using namespace cv::cuda;

/*
Detect the centre of red objects and mark these as pixels on a black frame
trackedFrame - A black frame with single red pixels representing the centre of red detected objects
cleanFrame - A frame of red and black, where red represents particularly red parts of the image
*/
__global__ void detectObjectKernel(cv::cuda::GpuMat trackedFrame, cv::cuda::GpuMat cleanFrame)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if ((threadId < X * Y) && ((threadId/X)%10 == 0) && ((threadId % X) % 10 == 0))
	{
		int row = threadId / X;
		int column = threadId % X;
		uint8_t pixelGClean = cleanFrame.data[(row*cleanFrame.step) + column * 3 + 1];
		if (pixelGClean == 255) {
			int maxX = column;
			int maxY = row;
			int minX = column;
			int minY = row;
			//Current array size is unstable, only works up until half the pixels being red (displayed as green), but saves memory and
			//it's expected there's effort to be as few red pixels as possible
			int pixelList[(X * Y) / 200];
			bool pixelUsed[X / 10][Y / 10] = { 0 };
			pixelList[0] = X * row + column;
			int listLength = 1;
			uint8_t pixelGcleanTest;
			int x, y, xTemp, yTemp, temp;
			while (listLength > 0) {
				x = pixelList[0] % X;
				y = pixelList[0] / X;
				if (!pixelUsed[x / 10][y / 10]) {
					if (x < X - 10) {
						pixelGcleanTest = cleanFrame.data[y*cleanFrame.step + (x + 10) * 3 + 1];
						if ((pixelGcleanTest == 255) && (!pixelUsed[(x + 10) / 10][y / 10])) {
							pixelList[listLength] = X * y + x + 10;
							listLength++;
							if (x + 10 > maxX) {
								maxX = maxX + 10;
							}
						}
					}
					if (x > 9) {
						pixelGcleanTest = cleanFrame.data[y*cleanFrame.step + (x - 10) * 3 + 1];
						if ((pixelGcleanTest == 255) && (!pixelUsed[(x - 10) / 10][y / 10])) {
							pixelList[listLength] = X * y + x - 10;
							listLength++;
							if (x - 10 < minX) {
								minX = minX - 10;
							}
						}
					}
					if (y < Y - 10) {
						pixelGcleanTest = cleanFrame.data[(y + 10)*cleanFrame.step + x * 3 + 1];
						if ((pixelGcleanTest == 255) && (!pixelUsed[x / 10][(y + 10) / 10])) {
							pixelList[listLength] = X * (y + 10) + x;
							listLength++;
							if (y + 10 > maxY) {
								maxY = maxY + 10;
							}
						}
					}
					if (y > 9) {
						pixelGcleanTest = cleanFrame.data[(y - 10)*cleanFrame.step + x * 3 + 1];
						if ((pixelGcleanTest == 255) && (!pixelUsed[x / 10][(y - 10) / 10])) {
							pixelList[listLength] = X * (y - 10) + x;
							listLength++;
							if (y - 10 < minY) {
								minY = minY - 10;
							}
						}
					}
				}
				pixelUsed[x / 10][y / 10] = true;
				temp = pixelList[listLength - 1];
				pixelList[listLength - 1] = pixelList[0];
				pixelList[0] = temp;
				listLength--;
			}
			int centreX = (maxX - minX) / 2 + minX;
			int centreY = (maxY - minY) / 2 + minY;
			trackedFrame.data[(centreY*trackedFrame.step) + centreX * 3 + 2] = 255;
		}
	}
}

/*
Erode the red image to restore closer to original object sizes
out - output of the eroded frame, red objects on a black background
dilatedFrame - input of an already dilated frame, red objects on a black background
*/
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
			for (int i = -2; i < 3; i++)
			{
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

/*
Dilate the red object on black background frame, to fill gaps inside the objects, if any are present
out - out of the dilated frame, in the form of red objects on a black background
redFrame - a frame with all of the detected red areas of the image highlighted, on a black background
*/
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

/*
Identify the areas of an image that are particularly red, so that the markers on the match can be detected
out - output, a frame with all of the areas that pass a "redness" threshold as red and the rest as black
*/
__global__ void getRedKernel(cv::cuda::GpuMat out)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < X * Y)
	{
		int row = threadId / X;
		int column = threadId % X;
		//BGR pixel values
		uint8_t pixelB = out.data[(row*out.step) + column * 3];
		uint8_t pixelG = out.data[(row*out.step) + column * 3 + 1];
		uint8_t pixelR = out.data[(row*out.step) + column * 3 + 2];
		if ((pixelR > 80) && (pixelB < 10) && (pixelG < 10))
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

/*
Create a black mat image
out - an entirely black frame
*/
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

/*
Identify where the match tip is in the image, based on the tracking dot centres of red areas and their relationship with each
other. The expectation is that there must be three tracking locations in a straight line, where the ratio between 2 locations
compared to another 1 with a shared location is approximately 1.5.
trackingLocations - The locations of the red tracking dots in the frame, that represent the center of red areas
matchTip - The output, an x-y coordinate for the determined match tip in the frame
*/
void getMatchLocation(std::set<int> *trackingLocations, int *matchTip){
	matchTip[0] = -1;
	matchTip[1] = -1;
	int trackA[2];
	int trackB[2];
	int trackC[2];
	int a[2];
	int b[2];
	double aMagnitude;
	double bMagnitude;
	double ratio;
	double oldRatio = 0.0;
	double dotProduct;
	//iterate over set to find all 3 location combinations, and find the most likely one to be the matchstick
	for (auto i : *trackingLocations) {
		for (auto j : *trackingLocations) {
			if (i != j) {
				for (auto k : *trackingLocations) {
					if ((i != k) && (j != k)) {
						trackA[0] = i % X;
						trackA[1] = i / X;
						trackB[0] = j % X;
						trackB[1] = j / X;
						trackC[0] = k % X;
						trackC[1] = k / X;
						a[0] = trackB[0] - trackA[0];
						a[1] = trackB[1] - trackA[1];
						b[0] = trackC[0] - trackB[0];
						b[1] = trackC[1] - trackB[1];
						aMagnitude = sqrt(a[0] * a[0] + a[1] * a[1]);
						bMagnitude = sqrt(b[0] * b[0] + b[1] * b[1]);
						dotProduct = (a[0] * b[0] + a[1] * b[1]) / (aMagnitude * bMagnitude);	
						if (dotProduct > 0.99) {
							ratio = bMagnitude / aMagnitude;
							//test if ratio is close to 1.5, as this is approximately the expected ratio
							if ((1.35 < ratio) && (ratio < 1.65) && (abs(1.6 - ratio) < abs(1.6 - oldRatio))){
								memcpy(matchTip, trackC, sizeof(int)*2);
								matchTip[0] += b[0] / 10;
								matchTip[1] += b[1] / 10;
								oldRatio = ratio;
							}
						}
					}
				}
			}			
		}
	}
}

/*
The main function for tracking the tip of the match. Take an input frame from the webcam and find the location where flame particles
should be emitted from, as an x-y coordinate of the match tip.
frame - The frame taken from the webcam
tip - The pointer to the memory location to store the tip coordinates
*/
void track(Mat frame, int *tip) {
	int threadCount = 1024;
	int blocks = (X * Y - 1) / threadCount + 1;
	if (blocks == 1)
	{
		threadCount = X * Y;
	}
	//Set up device variables
	uint8_t *d_outPtr;
	cv::cuda::GpuMat d_outFrame;
	d_outFrame.upload(frame);

	//Allocate device memory
	cudaMalloc((void **)&d_outPtr, d_outFrame.rows*d_outFrame.step);
	cudaMemcpyAsync(d_outPtr, d_outFrame.ptr<uint8_t>(), d_outFrame.rows*d_outFrame.step, cudaMemcpyDeviceToDevice);
	getRedKernel << <blocks, threadCount >> > (d_outFrame);

	uint8_t *d_dilatedPtr;
	cv::cuda::GpuMat d_dilatedFrame;
	d_outFrame.copyTo(d_dilatedFrame);

	//Allocate new device memory
	cudaMalloc((void**)&d_dilatedPtr, d_dilatedFrame.rows*d_dilatedFrame.step);
	cudaMemcpyAsync(d_dilatedPtr, d_dilatedFrame.ptr<uint8_t>(), d_dilatedFrame.rows*d_dilatedFrame.step, cudaMemcpyDeviceToDevice);

	dilateKernel << <blocks, threadCount >> > (d_dilatedFrame, d_outFrame);
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
	erodeKernel << <blocks, threadCount >> > (d_erodedFrame, d_dilatedFrame);
	//Free dilatedFrame pointer device memory
	cudaFree(d_dilatedPtr);
	d_dilatedFrame.release();

	uint8_t *d_trackedFramePtr;
	cv::cuda::GpuMat d_trackedFrame;
	//declare a totally black image matrix
	Mat newFrame(Y, X, CV_8UC3, cv::Scalar(0, 0, 0));
	d_trackedFrame.upload(newFrame);
	//Allocate new device memory
	cudaMalloc((void**)&d_trackedFramePtr, d_trackedFrame.rows*d_trackedFrame.step);
	cudaMemcpyAsync(d_trackedFramePtr, d_trackedFrame.ptr<uint8_t>(), d_trackedFrame.rows*d_trackedFrame.step, cudaMemcpyDeviceToDevice);
	
	detectObjectKernel<<<blocks, threadCount>>>(d_trackedFrame, d_erodedFrame);

	//Free erodedFrame pointer device memory
	cudaFree(d_erodedPtr);
	d_erodedFrame.release();

	cudaError_t error2 = cudaGetLastError();
	if (error2 != cudaSuccess) {
		printf("2. Error: %s\n", cudaGetErrorString(error2));
	}
	Mat trackedFrame;
	d_trackedFrame.download(trackedFrame);
	std::set<int> *trackingLocations = new std::set<int>[(X / 20)*(Y / 20)];
	for (int i = 0; i < Y; i = i + 5) {
		for (int j = 0; j < X; j = j + 5) {
			if (trackedFrame.data[(i*trackedFrame.step) + j * 3 + 2] == 255) {
				int thisPixel = i * X + j;
				trackingLocations[0].insert(thisPixel);
			}
		}
	}

	//Free the tracked frame from device memory
	cudaFree(d_trackedFramePtr);
	d_trackedFrame.release();
	trackedFrame.release();

	getMatchLocation(trackingLocations, tip);
	//uncomment for adding tracking marker to the frame
	/*
	if ((tip[0] > 0) && (tip[0] < X - 1) && (tip[0] > 0) && (tip[1] < Y - 1) && (tip[1] > 0)) {
		frame.data[tip[1] * frame.step + tip[0] * 3] = 0;
		frame.data[tip[1] * frame.step + tip[0] * 3 + 1] = 255;
		frame.data[tip[1] * frame.step + tip[0] * 3 + 2] = 0;
		frame.data[(tip[1] + 1) * frame.step + tip[0] * 3] = 0;
		frame.data[(tip[1] + 1) * frame.step + tip[0] * 3 + 1] = 255;
		frame.data[(tip[1] + 1) * frame.step + tip[0] * 3 + 2] = 0;
		frame.data[(tip[1] - 1) * frame.step + tip[0] * 3] = 0;
		frame.data[(tip[1] - 1) * frame.step + tip[0] * 3 + 1] = 255;
		frame.data[(tip[1] - 1) * frame.step + tip[0] * 3 + 2] = 0;
		frame.data[tip[1] * frame.step + (tip[0] + 1) * 3] = 0;
		frame.data[tip[1] * frame.step + (tip[0] + 1) * 3 + 1] = 255;
		frame.data[tip[1] * frame.step + (tip[0] + 1) * 3 + 2] = 0;
		frame.data[(tip[1] + 1) * frame.step + (tip[0] + 1) * 3] = 0;
		frame.data[(tip[1] + 1) * frame.step + (tip[0] + 1) * 3 + 1] = 255;
		frame.data[(tip[1] + 1) * frame.step + (tip[0] + 1) * 3 + 2] = 0;
		frame.data[(tip[1] - 1) * frame.step + (tip[0] + 1) * 3] = 0;
		frame.data[(tip[1] - 1) * frame.step + (tip[0] + 1) * 3 + 1] = 255;
		frame.data[(tip[1] - 1) * frame.step + (tip[0] + 1) * 3 + 2] = 0;
		frame.data[tip[1] * frame.step + (tip[0] - 1) * 3] = 0;
		frame.data[tip[1] * frame.step + (tip[0] - 1) * 3 + 1] = 255;
		frame.data[tip[1] * frame.step + (tip[0] - 1) * 3 + 2] = 0;
		frame.data[(tip[1] + 1) * frame.step + (tip[0] - 1) * 3] = 0;
		frame.data[(tip[1] + 1) * frame.step + (tip[0] - 1) * 3 + 1] = 255;
		frame.data[(tip[1] + 1) * frame.step + (tip[0] - 1) * 3 + 2] = 0;
		frame.data[(tip[1] - 1) * frame.step + (tip[0] - 1) * 3] = 0;
		frame.data[(tip[1] - 1) * frame.step + (tip[0] - 1) * 3 + 1] = 255;
		frame.data[(tip[1] - 1) * frame.step + (tip[0] - 1) * 3 + 2] = 0;
	}
	*/
	delete [] trackingLocations;
}