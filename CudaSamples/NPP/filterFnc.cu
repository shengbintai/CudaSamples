#include "filterFnc.cuh"

/*
* Summary:				计算区域均值
* @param	d_src:		输入图像指针, 指向一个存储在 GPU 设备内存中的单通道 8-bit 图像
* @param	d_sum:		输出结果指针，指向一个存储在 GPU 设备内存中的 8-bit 数据, 用来存储图像和
* @param	height:		图像高度
* @param	width:		图像宽度
*/
void calcFltBoxAvg(const u8* d_src, u8* d_dst, int height, int width)
{
	// source image line step
	int nSrcStep = width * sizeof(u8);

	// ROI
	NppiSize oSrcSize;
	oSrcSize.height = height;
	oSrcSize.width = width;
	NppiPoint oSrcOffset = { 0, 0 };

	NppiSize oSizeROI = { oSrcSize.width , oSrcSize.height };

	//选择区域求和的尺寸大小
	NppiSize oMaskSize = { 3, 3 };
	NppiPoint oAnchor = { oMaskSize.width / 2, oMaskSize.height / 2 };

	CHECKNPP(nppiFilterBoxBorder_8u_C1R(d_src, nSrcStep, oSrcSize, oSrcOffset, d_dst, nSrcStep, oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE));

}
using namespace cv;
void calcFltConv(const u8* d_src, u8* d_dst, int height, int width)
{
	// source image line step
	int nSrcStep = width * sizeof(u8);

	// ROI
	NppiSize oSrcSize;
	oSrcSize.height = height;
	oSrcSize.width = width;
	NppiPoint oSrcOffset = { 0, 0 };

	NppiSize oSizeROI = { oSrcSize.width , oSrcSize.height };

	Mat a=Mat::ones(3,3,CV_32SC1);
	int stepBytes=4*3;
	Npp32s *kernel;// = nppiMalloc_32s_C1(4, 4, &stepBytes);
	CHECKCUDA(cudaMalloc((void **)&kernel, 3 *3 * sizeof(int)));
	CHECKCUDA(cudaMemcpy(kernel, a.ptr<int>(0), 3 * 3 * sizeof(int),
		cudaMemcpyHostToDevice));

	CHECKCUDA(cudaMemcpy(a.ptr<int>(0), kernel, 3 * 3 * sizeof(int),cudaMemcpyDeviceToHost));
	//选择区域求和的尺寸大小
	NppiSize oKernelSize = { 3, 3 };
	NppiPoint oAnchor = { oKernelSize.width / 2, oKernelSize.height / 2 };

	CHECKNPP(nppiFilterBorder_8u_C1R(d_src, nSrcStep, oSrcSize, oSrcOffset, d_dst, nSrcStep, oSizeROI, kernel, oKernelSize, oAnchor,2, NPP_BORDER_REPLICATE));

}