#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
using namespace std;
#include <opencv2/core.hpp>
using namespace cv;
#include <opencv2/cudaarithm.hpp>
using namespace cv::cuda;

__global__ void trackKernel(int *c, const int *a, const int *b)
{
	//detect end
	//determine orientation
	//determine distance
	//have internal representation of it's position in 3D
	//draw particles in 3D space
	//move particles with physics based on the match's movement
}

int main()
{
	//test code from Get started with OpenCV CUDA cpp
	printShortCudaDeviceInfo(getDevice());
	int cuda_devices_number = getCudaEnabledDeviceCount();
	cout << "CUDA Device(s) Number: " << cuda_devices_number << endl;
	DeviceInfo _deviceInfo;
	bool _isd_evice_compatible = _deviceInfo.isCompatible();
	cout << "CUDA Device(s) Compatible: " << _isd_evice_compatible << endl;
	return 0;
}