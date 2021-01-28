#include <stdio.h>
#include "math.h"
#include <string>

// Image IO
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>

#include "types.cuh"
// Output generated from Teg
#include "perlin.cuh"


// End Temporary placeholder.
#define ALPHA_COLOR 0.001
#define ALPHA_VERTEX 0.08

int main(int argc, char** argv)
{
    if (argc != 6) {
        std::cout << "Usage: ./render_perlin <res-x> <res-y> <grid-nx> <grid-ny> <seed>" << std::endl;
        exit(1);
    }

    std::stringstream ss_rx(argv[1]);
    std::stringstream ss_ry(argv[2]);
    std::stringstream ss_nx(argv[3]);
    std::stringstream ss_ny(argv[4]);
    std::stringstream ss_seed(argv[5]);

    int nx;
    int ny;
    int rx;
    int ry;
    int seed;

    ss_nx >> nx;
    ss_ny >> ny;
    ss_rx >> rx;
    ss_ry >> ry;
    ss_seed >> seed;

    if (rx % nx || ry % ny) {
        std::cout << "Resoution must be a perfect multiple of grid dimensions" << std::endl;
        exit(1);
    }

    std::cout << "Rendering perlin noise, resolution: " << rx << "x" << ry << ", grid: " << nx << "x" << ny << std::endl;

    float* texture;
    Color* pcolor;
    Color* ncolor;
    DColor* d_pcolor;
    DColor* d_ncolor;

    float* threshold;
    float* d_threshold;

    Grid* grid;
    DGrid* d_grid;

    char* triangle_bimage = (char*) malloc(rx * ry * 3);

    build_grid(nx, ny, &grid, seed);
    build_d_grid(grid, &d_grid);
    //build_image(&texture, rx, ry);

    cudaMallocManaged(&texture, sizeof(float) * rx * ry * 3);
    cudaMallocManaged(&pcolor, sizeof(Color));
    cudaMallocManaged(&ncolor, sizeof(Color));
    cudaMallocManaged(&threshold, sizeof(float));

    cudaMallocManaged(&d_pcolor, sizeof(DColor));
    cudaMallocManaged(&d_ncolor, sizeof(DColor));
    cudaMallocManaged(&d_threshold, sizeof(float));

    // Baby blue.
    pcolor->b = 0.5373;
    pcolor->g = 0.8118;
    pcolor->r = 0.9412;

    // Baby pink.
    ncolor->b = 0.9569;
    ncolor->g = 0.7608;
    ncolor->r = 0.7608;

    *threshold = 0;
    *d_threshold = 0;

    std::cout << "Built structures" << std::endl;

    int num_jobs = rx * ry;
    int* qids;
    int* pids;

    cudaMallocManaged(&qids,     num_jobs * sizeof(int));
    cudaMallocManaged(&pids,     num_jobs * sizeof(int));

    for(int x = 0; x < rx; x++)
        for(int y = 0; y < ry; y++)
            for(int c = 0; c < 3; c++)
                texture[(y * rx + x) * 3 + c] = 0.f;

    int scale_x = rx / nx;
    int scale_y = ry / ny;
    int job_count = 0;
    for(int x = 0; x < rx; x++) {
        for(int y = 0; y < ry; y++) {
            pids[job_count] = y * rx + x;
            qids[job_count] = (y / scale_y) * nx + (x / scale_x);
            job_count ++;
        }
    }

    //std::cout << type2str(image.type()) << std::endl;

    std::cout << "Jobs: " << job_count;
    std::cout << "Scale: " << scale_x << ", " << scale_y << std::endl;
    // Render noise.
    /*
    perlin_threshold_kernel<<<(job_count / 256) + 1, 256>>>(
                qids,
                pids,
                job_count,
                grid,
                pcolor,
                ncolor,
                threshold,
                texture,
                rx, ry);
    */
    perlin_threshold_kernel<<<(job_count / 256) + 1, 256>>>(
                        qids,
                        pids,
                        job_count,
                        grid,
                        pcolor,
                        ncolor,
                        threshold,
                        texture,
                        rx, ry);

    cudaDeviceSynchronize();

    for(int idx = 0; idx < rx * ry * 3; idx ++){
        // float val = (texture[idx] + 1) * 0.5;
        float val = texture[idx];
        int _val = (int)(val * 256);
        // std::cout << "VAL: " << texture[idx] << std::endl;
        triangle_bimage[idx] = (char) ((_val < 0) ? 0 : (_val > 255 ? 255 : _val));
    }

    std::stringstream ss;
    ss << "noise-" << seed << ".png";
    cv::imwrite(ss.str(), cv::Mat(rx, ry, CV_8UC3, triangle_bimage));

    for(int x = 0; x < rx; x++)
        for(int y = 0; y < ry; y++)
            for(int c = 0; c < 3; c++)
                texture[(y * rx + x) * 3 + c] = 0.f;


    /*for (int i = 0; i < 50; i++)
        for (int j = 0; j < 50; j++){
            float f0 = d_tcolors[(i * 50 + j) * 3 + 0];
            float f1 = d_tcolors[(i * 50 + j) * 3 + 1];
            float f2 = d_tcolors[(i * 50 + j) * 3 + 2];
            if (f0 != 0 || f1 != 0 || f2 != 0)
                std::cout << f0 << ", " << f1 << ", " << f2 << std::endl;
        }

    for (int i = 0; i < 50; i++)
        for (int j = 0; j < 50; j++){
            float f0 = d_vertices[(i * 50 + j) * 2 + 0];
            float f1 = d_vertices[(i * 50 + j) * 2 + 1];
            if (f0 != 0 || f1 != 0)
                std::cout << f0 << ", " << f1 << std::endl;
        }*/

}