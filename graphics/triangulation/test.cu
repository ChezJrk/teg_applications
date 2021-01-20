#include <stdio.h>
#include "math.h"
#include <string>

// Image IO
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>

#include "common.cuh"
#include "triangle.cuh"
// Output generated from Teg
#include "quadratic.cuh"
//#include "linear.cuh"

int main(int argc, char** argv)
{

    
    TriMesh* mesh;

    cudaMallocManaged(&mesh, sizeof(TriMesh));

    mesh->num_vertices = 3;
    cudaMallocManaged(&(mesh->vertices), sizeof(Vertex) * mesh->num_vertices);

    cudaMallocManaged(&(mesh->weights), sizeof(float) * 3);

    mesh->num_triangles = 1;
    cudaMallocManaged(&(mesh->triangles), sizeof(Triangle) * mesh->num_triangles);

    mesh->vertices[0] = Vertex{100, 100};
    mesh->vertices[1] = Vertex{400, 100};
    mesh->vertices[2] = Vertex{100, 400};

    mesh->triangles[0] = Triangle{0, 2, 1};

    
    QuadraticFragment* colors;
    cudaMallocManaged(&colors, sizeof(QuadraticFragment) * 1);
    colors[0] = QuadraticFragment{Color{0, 1, 0}, Color{0, 0, 1}, Color{1, 0, 0},
                                  Color{1, 0, 1}, Color{1, 1, 0}, Color{0, 1, 1}};
    
    /*
    LinearFragment* colors;
    cudaMallocManaged(&colors, sizeof(LinearFragment) * 1);
    colors[0] = LinearFragment{Color{0, 1, 0}, Color{0, 0, 1}, Color{1, 0, 0}};
    */


    int h = 512;
    int w = 512;

    float* triangle_image;
    char* triangle_bimage = (char*) malloc(w * h * 3);
    cudaMallocManaged(&triangle_image, sizeof(float) * w * h * 3);

    int* tids;
    int* pids;

    int num_jobs = w * h;
    cudaMallocManaged(&tids,     num_jobs * sizeof(int));
    cudaMallocManaged(&pids,     num_jobs * sizeof(int));
    
    for (int i = 0; i < h * w; i++) {
        tids[i] = 0;
        pids[i] = i;
    }

    set_zero<float><<<((w * h * 3) / 256) + 1, 256>>>(triangle_image, (w * h * 3) );

    // Render triangles to image.
    quadratic_integral_kernel<<<(num_jobs / 256) + 1, 256>>>(
                tids,
                pids,
                num_jobs,
                mesh,
                colors,
                triangle_image,
                w, h);

    cudaDeviceSynchronize();

    for(int idx = 0; idx < w * h * 3; idx ++){
        int _val = (int)(triangle_image[idx] * 256);
        triangle_bimage[idx] = (char) ((_val < 0) ? 0 : (_val > 255 ? 255 : _val));
    }

    std::stringstream ss;
    ss << "test.png";
    cv::imwrite(ss.str(), cv::Mat(w, h, CV_8UC3, triangle_bimage));


    cudaFree(triangle_image);
    cudaFree(tids);
    cudaFree(pids);

}