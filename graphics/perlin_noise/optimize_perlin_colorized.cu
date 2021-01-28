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
#include "perlin_colorized.cuh"


// End Temporary placeholder.
#define ALPHA_COLOR 0.000001
#define ALPHA_VERTEX 0.005
#define ALPHA_THRESHOLD 10e-8

int main(int argc, char** argv)
{
    if (argc != 6) {
        std::cout << "Usage: ./optimize_perlin <image> <grid-nx> <grid-ny> <seed> <lrate>" << std::endl;
        exit(1);
    }

    std::stringstream ss_nx(argv[2]);
    std::stringstream ss_ny(argv[3]);
    std::stringstream ss_seed(argv[4]);
    std::stringstream ss_learning_rate(argv[5]);

    int nx;
    int ny;
    int seed;
    float alpha_vertex;

    ss_nx >> nx;
    ss_ny >> ny;
    ss_seed >> seed;
    ss_learning_rate >> alpha_vertex;

    // Load an image.
    cv::Mat image;
    image = cv::imread(argv[1], cv::IMREAD_COLOR);

    if( !image.data ) {
        std::cout <<  "Could not open or find the image" << std::endl;
        return -1;
    }

    int rx = image.rows;
    int ry = image.cols;

    Image* target_image;
    cudaMallocManaged(&target_image, sizeof(Image));

    target_image->rows = image.rows;
    target_image->cols = image.cols;
    cudaMallocManaged(&(target_image->colors), sizeof(Color) * image.rows * image.cols);

    // Load image data.
    for(int i = 0; i < image.rows; i++)
        for(int j = 0; j < image.cols; j++){
            cv::Vec3b v = image.at<cv::Vec3b>(i, j);
            target_image->colors[(image.rows * j + i)].r = ((float)v[0]) / 255.0;//*(image.data + idx + 0);
            target_image->colors[(image.rows * j + i)].g = ((float)v[1]) / 255.0;//*(image.data + idx + 1);
            target_image->colors[(image.rows * j + i)].b = ((float)v[2]) / 255.0;//*(image.data + idx + 2);
        }

    std::cout << "Fitting " << rx << "x" << ry << " image" << std::endl;

    if (rx % nx || ry % ny) {
        std::cout << "Resoution must be a perfect multiple of grid dimensions" << std::endl;
        exit(1);
    }

    float* texture;
    //Color* pcolor;
    Color* ncolor;
    //DColor* d_pcolor;
    DColor* d_ncolor;

    float* threshold;
    float* d_threshold;

    Grid* grid;
    DGrid* d_grid;

    char* bimage = (char*) malloc(rx * ry * 3);

    build_grid(nx, ny, &grid, seed);
    build_d_grid(grid, &d_grid);
    //build_image(&texture, rx, ry);

    cudaMallocManaged(&texture, sizeof(float) * rx * ry * 3);
    //cudaMallocManaged(&pcolor, sizeof(Color));
    cudaMallocManaged(&ncolor, sizeof(Color));
    cudaMallocManaged(&threshold, sizeof(float));

    //cudaMallocManaged(&d_pcolor, sizeof(DColor));
    cudaMallocManaged(&d_ncolor, sizeof(DColor));
    cudaMallocManaged(&d_threshold, sizeof(float));

    // Baby blue.
    /*
    pcolor->b = 0.5373;
    pcolor->g = 0.8118;
    pcolor->r = 0.9412;
    */

    //pcolor->b = 0.3373;
    //pcolor->g = 0.6118;
    //pcolor->r = 0.7112;

    // Baby pink.
    /*
    ncolor->b = 0.9569;
    ncolor->g = 0.7608;
    ncolor->r = 0.7608;
    */
    ncolor->b = 0.8569;
    ncolor->g = 0.8569;
    ncolor->r = 0.8569;

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
    for (int iter = 0; iter < 300; iter ++){
        std::cout << "Iteration " << iter << std::endl;
        
        /*
        for(int x = 0; x < rx; x++)
            for(int y = 0; y < ry; y++)
                for(int c = 0; c < 3; c++)
                    texture[(y * rx + x) * 3 + c] = 0.f;
        */
        
        fast_zero<float>(texture, rx * ry * 3);
        perlin_colorized_threshold_kernel<<<(job_count / 256) + 1, 256>>>(
                            qids,
                            pids,
                            job_count,
                            grid,
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
            bimage[idx] = (char) ((_val < 0) ? 0 : (_val > 255 ? 255 : _val));
        }

        std::stringstream ss;
        ss << "noise-" << iter << ".png";
        cv::imwrite(ss.str(), cv::Mat(rx, ry, CV_8UC3, bimage));

        // Zero derivatives before accumulating derivs.
        *d_threshold = 0.f;
        *d_ncolor = 0.f;
        fast_zero<DVec2D>(d_grid->d_vecs, (d_grid->rows + 1) * (d_grid->cols + 1));
        fast_zero<DColor>(d_grid->d_colors, (d_grid->rows + 1) * (d_grid->cols + 1));

        // Compute and accumulate derivs.
        perlin_colorized_threshold_deriv<<<(job_count / 256) + 1, 256>>>(
                            qids,
                            pids,
                            job_count,
                            target_image,
                            grid,
                            d_grid,
                            ncolor,
                            d_ncolor,
                            threshold,
                            d_threshold,
                            rx, ry);

        cudaDeviceSynchronize();
        
        dim3 dimBlock(32, 32);
        dim3 dimGrid(((grid->rows + 1) / 32) + 1, ((grid->cols + 1) / 32) + 1);
        update_grid_positions<<<dimGrid, dimBlock>>>(grid, d_grid, alpha_vertex, ALPHA_COLOR * (grid->rows * grid->cols));

        /*
        pcolor->r = pcolor->r - d_pcolor->r * ALPHA_COLOR;
        pcolor->g = pcolor->g - d_pcolor->g * ALPHA_COLOR;
        pcolor->b = pcolor->b - d_pcolor->b * ALPHA_COLOR;
        */

        //ncolor->r = ncolor->r - d_ncolor->r * ALPHA_COLOR;
        //ncolor->g = ncolor->g - d_ncolor->g * ALPHA_COLOR;
        //ncolor->b = ncolor->b - d_ncolor->b * ALPHA_COLOR;

        //*threshold = *threshold - (*d_threshold) * ALPHA_THRESHOLD;

    }


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