#include "filterFnc.cuh"

/*
* Summary:				���������ֵ
* @param	d_src:		����ͼ��ָ��, ָ��һ���洢�� GPU �豸�ڴ��еĵ�ͨ�� 8-bit ͼ��
* @param	d_sum:		������ָ�룬ָ��һ���洢�� GPU �豸�ڴ��е� 8-bit ����, �����洢ͼ���
* @param	height:		ͼ��߶�
* @param	width:		ͼ����
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

	//ѡ��������͵ĳߴ��С
	NppiSize oMaskSize = { 3, 3 };
	NppiPoint oAnchor = { oMaskSize.width / 2, oMaskSize.height / 2 };

	CHECKNPP(nppiFilterBoxBorder_8u_C1R(d_src, nSrcStep, oSrcSize, oSrcOffset, d_dst, nSrcStep, oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE));

}