#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <npp.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>
#include <helper_cuda.h>
#include <nppdefs.h>

#include <stdio.h>
#include <iostream>
using u16 = unsigned short;
using u8 = unsigned char;
using namespace std;

#ifndef CHECKNPP
/*
* Summary: 检查 npp 库函数是否调用成功
*/
#define CHECKNPP(S)																	\
do {																				\
		NppStatus eStatusNPP;														\
		eStatusNPP = S;																\
		if (eStatusNPP != NPP_SUCCESS){												\
			printf("ERROR: %s:%d,", __FILE__, __LINE__);							\
			std::cout << "NPP_CHECK_NPP - eStatusNPP = " <<							\
			_cudaGetErrorEnum(eStatusNPP) << "("<< eStatusNPP << ")" << std::endl;	\
		}																			\
} while (false)
#endif // CHECKNPP

#ifndef CHECKCUDA
/*
* Summary: 检查 CUDA 函数是否调用成功
*/
#define CHECKCUDA(call)                                                   \
do{                                                                       \
	const cudaError_t error = call;                                       \
	if(error != cudaSuccess)                                              \
	{                                                                     \
		printf("ERROR: %s:%d,", __FILE__, __LINE__);                      \
		printf("code:%d, reason:%s\n", error, cudaGetErrorString(error)); \
		exit(1);                                                          \
	}                                                                     \
}while(0)
#endif // CHECKCUDA


void calcFltBoxAvg(const u8* d_src, u8* d_dst, int height, int width);