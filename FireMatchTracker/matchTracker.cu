#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "matchTracker.h"

__global__ void trackKernel(int *out, Mat frame, int x, int y)
{
	//detect end
	//determine orientation
	//determine distance
	//have internal representation of it's position in 3D
	//draw particles in 3D space
	//move particles with physics based on the match's movement

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId < x * y)
	{
		int thisX = ;
		int thisY = ;
		Vec3b pixel = frame.at<Vec3b>(thisX, thisY);
	}
}

void track(Mat frame) {
	Vec3b pixel = frame.at<Vec3b>(0, 0);
	//BGR pixel values
	printf("%d %d %d\n", pixel[0], pixel[1], pixel[2]);
	int threadCount = 1024;
	int x = 1280;
	int y = 720;
	int blocks = (x * y - 1) / threadCount + 1;
	//trackKernel<<<, >>>();
}