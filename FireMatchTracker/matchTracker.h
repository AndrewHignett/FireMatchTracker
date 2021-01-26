#pragma once
#include <opencv2/core.hpp>
#include <opencv2/cudafilters.hpp>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
using namespace cv::cuda;
using namespace cv;
#include <algorithm>

#ifndef _matchTracker_
#define _matchTracker_
/*
Define function template for the track function, for tracking the match tip
*/
void track(Mat frame, int *tip);

/*
Declared dimensions of the window captured through the webcam
*/
const int X = 1280;
const int Y = 720;
#endif