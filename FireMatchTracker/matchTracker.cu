#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "matchTracker.h"

__global__ void trackKernel(int *c, const int *a, const int *b)
{
	//detect end
	//determine orientation
	//determine distance
	//have internal representation of it's position in 3D
	//draw particles in 3D space
	//move particles with physics based on the match's movement
}

void track(Mat frame) {
	Vec3b pixel = frame.at<Vec3b>(0, 0);
	//BGR pixel values
	printf("%d %d %d\n", pixel[0], pixel[1], pixel[2]);
	//trackKernel<<<, >>>();
}