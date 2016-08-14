#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <omp.h>

#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;
using namespace cv::gpu;

#define N 4096

/*Kernel Sobel CUDA*/
__global__ void sobelKernel(unsigned char src[][N], unsigned char dest[][N])
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	for (; i < N; i += gridDim.x*blockDim.x) {
		for (; j < N; j += gridDim.y*blockDim.y) {
			short tmp = 0;
			short tmp2 = 0;
			
			if (i<0 || j<0 || N - 1<i || N - 1<j);
			else
			{
				/*convolucion con el Kernel vertical*/
				tmp -= src[i - 1][j - 1];
				tmp -= 2 * src[i][j - 1];
				tmp -= src[i + 1][j - 1];
				tmp += src[i - 1][j + 1];
				tmp += 2 * src[i][j + 1];
				tmp += src[i + 1][j + 1];
				/*convolucion con el kernel horizontal*/
				tmp2 -= src[i - 1][j - 1];
				tmp2 += src[i + 1][j - 1];
				tmp2 -= 2 * src[i - 1][j];
				tmp2 += 2 * src[i + 1][j];
				tmp2 -= src[i - 1][j + 1];
				tmp2 += src[i + 1][j + 1];
				/*suma de los resultados de la convolucion horizontal y vertical*/
				tmp = (short)((abs((int)tmp) + abs((int)tmp2)));
			}
			if (tmp < 0) {
				dest[i][j] = 0;
			}
			else if (tmp > 0xff) {
				dest[i][j] = 0xff;
			}
			else
				dest[i][j] = (unsigned char)tmp;
		}
	}	
}

void serialSobel(Mat src, Mat dest);
void ompSobel(Mat src, Mat dest);

int main()
{
	/**carga de la imagen con OpenCV**/
	Mat image;
	image = imread("mario.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat imageCuda;
	imageCuda = imread("mario.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat imageSerial;
	imageSerial = imread("mario.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat imageOmp;
	imageOmp = imread("mario.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	
	if (!image.data || !imageCuda.data || !imageSerial.data || !imageOmp.data)
	{
		printf("No image data \n");
		return -1;
	}
	namedWindow("Display Image", CV_WINDOW_NORMAL);
	imshow("Display Image", image);
	
	/*transcripcion de la imagen a un array para enviar a Cuda*/
	unsigned char *C = (unsigned char*)malloc(N*N*sizeof(unsigned char));
	for (size_t i = 0; i < image.rows; i++)	{
		for (size_t j = 0; j < image.cols; j++)	{
			C[i*image.rows + j] = image.at<uchar>(i, j);
		}
	}

	unsigned char(*pC)[N];
	unsigned char(*pD)[N];
	/*variables para tomar tiempos de Cuda*/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	/*Asignacion de memoria en Cuda*/
	cudaMalloc((void**)&pC, (N*N)*sizeof(unsigned char));
	cudaMalloc((void**)&pD, (N*N)*sizeof(unsigned char));
	/*envio de datos al device*/
	cudaMemcpy(pC, C, (N*N)*sizeof(unsigned char), cudaMemcpyHostToDevice);
	dim3 numBlocks(128, 128);
	dim3 threadsPerBlock(32, 32);
	cudaEventRecord(start);
	/*llamada al kernel sobel de Cuda*/
	sobelKernel << <numBlocks, threadsPerBlock >> >(pC,pD);
	cudaEventRecord(stop);
	/*recuperacion de datos procesados del device*/
	cudaMemcpy(C, pD, (N*N)*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Elapsed time CUDA : %f ms\n", milliseconds);
	printf("Effective Bandwidth (GB/s): %f\n\n", N *N * 4 * 3 / milliseconds / 1e6);
	/*transcripcion de los datos recibidos a imagen*/
	for (size_t i = 0; i < imageCuda.rows; i++)	{
		for (size_t j = 0; j < imageCuda.cols; j++)	{
			imageCuda.at<uchar>(i, j) = C[i*imageCuda.rows + j];
		}
	}
	/*display de la imagen generada por cuda*/
	namedWindow("CUDA Sobel", CV_WINDOW_NORMAL);
	imshow("CUDA Sobel", imageCuda);
	/*liberacion de memoria del device*/
	cudaFree(pC);
	cudaFree(pD);
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	/*ejecucion sobel secuencial y paralelo en cpu con openMP*/
	ompSobel(image, imageOmp);
	serialSobel(image, imageSerial);
	
	waitKey(0);
	return 0;
}

/*sobel secuancial*/
void serialSobel(Mat src, Mat dest){
	double t = (double)getTickCount();
	for (int i = 1; i < N - 1; i++) {
		for (int j = 1; j < N - 1; j++) {
			short tmp = 0;
			short tmp2 = 0;
			if (i<0 || j<0 || N - 1<i || N - 1<j);
			else
			{
				tmp -= src.at<uchar>(i - 1, j - 1);
				tmp -= 2 * src.at<uchar>(i, j - 1);
				tmp -= src.at<uchar>(i + 1, j - 1);
				tmp += src.at<uchar>(i - 1, j + 1);
				tmp += 2 * src.at<uchar>(i, j + 1);
				tmp += src.at<uchar>(i + 1, j + 1);
				tmp2 -= src.at<uchar>(i - 1, j - 1);
				tmp2 += src.at<uchar>(i + 1, j - 1);
				tmp2 -= 2 * src.at<uchar>(i - 1, j);
				tmp2 += 2 * src.at<uchar>(i + 1, j);
				tmp2 -= src.at<uchar>(i - 1, j + 1);
				tmp2 += src.at<uchar>(i + 1, j + 1);
				tmp = (short)((abs((int)tmp) + abs((int)tmp2)));
			}
			if (tmp < 0) {
				dest.at<uchar>(i, j) = 0;
			}
			else if (tmp > 0xff) {
				dest.at<uchar>(i, j) = 0xff;
			}
			else
				dest.at<uchar>(i, j) = (unsigned char)tmp;
		}
	}
	t = ((double)getTickCount() - t) / getTickFrequency();
	std::cout << "Elapsed time Serial: " << t*1000 <<" ms." << std::endl;
	namedWindow("Serial Sobel", CV_WINDOW_NORMAL);
	imshow("Serial Sobel", dest);
}

/*sobel con paralelismo CPU OpenMP*/
void ompSobel(Mat src, Mat dest){
	double t = (double)getTickCount();
	#pragma omp parallel for
	for (int i = 1; i < N - 1; i++) {
		for (int j = 1; j < N - 1; j++) {
			short tmp = 0;
			short tmp2 = 0;
			if (i<0 || j<0 || N - 1<i || N - 1<j);
			else
			{
				tmp -= src.at<uchar>(i - 1, j - 1);
				tmp -= 2 * src.at<uchar>(i, j - 1);
				tmp -= src.at<uchar>(i + 1, j - 1);
				tmp += src.at<uchar>(i - 1, j + 1);
				tmp += 2 * src.at<uchar>(i, j + 1);
				tmp += src.at<uchar>(i + 1, j + 1);
				tmp2 -= src.at<uchar>(i - 1, j - 1);
				tmp2 += src.at<uchar>(i + 1, j - 1);
				tmp2 -= 2 * src.at<uchar>(i - 1, j);
				tmp2 += 2 * src.at<uchar>(i + 1, j);
				tmp2 -= src.at<uchar>(i - 1, j + 1);
				tmp2 += src.at<uchar>(i + 1, j + 1);
				tmp = (short)((abs((int)tmp) + abs((int)tmp2)));
			}
			if (tmp < 0) {
				dest.at<uchar>(i, j) = 0;
			}
			else if (tmp > 0xff) {
				dest.at<uchar>(i, j) = 0xff;
			}
			else
				dest.at<uchar>(i, j) = (unsigned char)tmp;
		}
	}
	t = ((double)getTickCount() - t) / getTickFrequency();
	std::cout << "Elapsed time OMP: " << t * 1000 << " ms." << std::endl;
	namedWindow("Omp Sobel", CV_WINDOW_NORMAL);
	imshow("Omp Sobel", dest);
}