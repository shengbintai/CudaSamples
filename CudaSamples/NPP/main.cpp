#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "filterFnc.cuh"

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
	string path = "./Img/Lena.tif";
	Mat img = imread(path, IMREAD_UNCHANGED);
	Mat img_o(img.size(), img.type());
	int height = img.rows;
	int width = img.cols;

	u8 *d_src;
	u8 *d_dst;

	CHECKCUDA(cudaMalloc((void**)&d_src, height*width * sizeof(u8)));
	CHECKCUDA(cudaMalloc((void**)&d_dst, height*width * sizeof(u8)));
	CHECKCUDA(cudaMemcpy(d_src, img.ptr<u8>(0), height * width * sizeof(u8),
		cudaMemcpyHostToDevice));
	calcFltBoxAvg( d_src,  d_dst,  height,  width);
	CHECKCUDA(cudaMemcpy(img_o.ptr<u8>(0), d_dst, height * width * sizeof(u8),
		cudaMemcpyDeviceToHost));

	CHECKCUDA(cudaFree(d_src));
	CHECKCUDA(cudaFree(d_dst));

	return 0;
}

