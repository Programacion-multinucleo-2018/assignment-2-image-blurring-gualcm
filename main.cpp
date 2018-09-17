/* Gualberto Casas */
/* A00942270 */
/* https://en.wikipedia.org/wiki/Box_blur */
/* compile: command */

#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "omp.h"
#include <cuda_runtime.h>

using namespace std;

void blur_GPU_CUDA(cv::Mat& input, cv::Mat& output)
{
    printf("On development");
}

void blur_CPU_OMP(cv::Mat& input, cv::Mat& output)
{
    int i, j, tx, ty;
    int size = 9;
    #pragma omp parallel private(i, j, tx, ty) shared (input, output, size)
    {
        for (i = 0; i < input.rows; i++)
        {
            for (j = 0; j < input.cols; j++)
            {
                int uvx, uvy;
                int red = 0;
                int green = 0;
                int blue = 0;

                for (uvx = -1; uvx <= 1; uvx++)
                {
                    for (uvy = -1; uvy <= 1; uvy++)
                    {
                        tx = uvx + i;
                        ty = uvy + j;
                        if ((tx > 0) && (tx < input.rows) && (ty > 0) && (ty < input.cols))
                        {
                            red+= input.at<cv::Vec3b>(tx, ty)[0];
                            green+= input.at<cv::Vec3b>(tx, ty)[1];
                            blue+= input.at<cv::Vec3b>(tx, ty)[2];
                        }
                    }
                }

                output.at<cv::Vec3b>(i, j)[0] = (red / size);
                output.at<cv::Vec3b>(i, j)[1] = (green / size);
                output.at<cv::Vec3b>(i, j)[2] = (blue / size);
            }
        }
    }
}

void blur_CPU_no_threads(cv::Mat& input, cv::Mat& output)
{
    int i, j;
    int size = 9;

    for (i = 0; i < input.rows; i++)
    {
        for (j = 0; j < input.cols; j++)
        {
            int tx, ty, uvx, uvy;
            int red = 0;
            int green = 0;
            int blue = 0;

            for (uvx = -1; uvx <= 1; uvx++)
            {
                for (uvy = -1; uvy <= 1; uvy++)
                {
                    tx = uvx + i;
                    ty = uvy + j;

                    if ((tx > 0) && (tx < input.rows) && (ty > 0) && (ty < input.cols))
                    {
                        red += input.at<cv::Vec3b>(tx, ty)[0];
                        green += input.at<cv::Vec3b>(tx, ty)[1];
                        blue += input.at<cv::Vec3b>(tx, ty)[2];
                    }
                }
            }

            output.at<cv::Vec3b>(i, j)[0] = (red / size);
            output.at<cv::Vec3b>(i, j)[1] = (green / size);
            output.at<cv::Vec3b>(i, j)[2] = (blue / size);
        }
    } 
}

int main(int argc, char const *argv[])
{
    printf("Starting...\n");
    float average = 0.0;

    // Image Path
    string image;
    if (argc < 2) image = "image.jpg";
    else image = argv[1];

    // Read Image
    cv::Mat input = cv::imread(image, CV_LOAD_IMAGE_COLOR);

    // Output Image
    cv::Mat output1(input.rows, input.cols, CV_8UC3); // CPU NO THREADS
    cv::Mat output2(input.rows, input.cols, CV_8UC3); // CPU OMP
    cv::Mat output3(input.rows, input.cols, CV_8UC3); // GPU CUDA

    // Start timer CPU NO THREADS
    int i;
    for (i = 0; i < 20; i++)
    {
        auto start =  chrono::high_resolution_clock::now();
        blur_CPU_no_threads(input, output1);
        auto end =  chrono::high_resolution_clock::now();
        chrono::duration<float, std::milli> duration_ms = end - start;
        printf("Time passed (CPU NO THREADS): %f ms", duration_ms.count());
        average += duration_ms;
    }
    printf("Average: %f ms", average);
    average = 0.0;

    // Window Resizing
    /* namedWindow("Input", cv::WINDOW_NORMAL); */
    /* namedWindow("Output", cv::WINDOW_NORMAL); */

    // Display images
    /* imshow("Input", input); */
    /* imshow("Output", output); */

    // Start timer CPU OMP
    for (i = 0; i < 20; i++)
    {
        start =  chrono::high_resolution_clock::now();
        blur_CPU_OMP(input, output2);
        end =  chrono::high_resolution_clock::now();
        duration_ms = end - start;
        printf("Time passed (CPU OMP): %f ms", duration_ms.count());
        average += duration_ms;
    }
    printf("Average: %f ms", average);
    average = 0.0;

    // Start timer GPU CUDA
    for (i = 0; i < 20; i++)
    {
        start =  chrono::high_resolution_clock::now();
        blur_GPU_CUDA(input, output3);
        end =  chrono::high_resolution_clock::now();
        duration_ms = end - start;
        printf("Time passed (GPU CUDA): %f ms", duration_ms.count());
        average += duration_ms;
    }
    printf("Average: %f ms", average);

    return 0;
}
