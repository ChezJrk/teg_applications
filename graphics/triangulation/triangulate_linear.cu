#include <stdio.h>
#include "math.h"
#include <string>

// Image IO
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>
#include <fstream>

#include "common.cuh"
#include "triangle.cuh"
// Output generated from Teg
#include "linear.cuh"


// End Temporary placeholder.
//#define ALPHA_COLOR 0.001
//#define ALPHA_VERTEX 0.08

#define ALPHA_COLOR 0.3
#define ALPHA_VERTEX 10000

int main(int argc, char** argv)
{
    if (argc != 5) {
        std::cout << "Usage: ./triangulate_linear <image_file> <tri-grid nx> <tri-grid ny> <use-deltas:y/n>" << std::endl;
        exit(1);
    }

    std::stringstream ss_nx(argv[2]);
    std::stringstream ss_ny(argv[3]);
    std::stringstream ss_delta(argv[4]);

    int nx;
    int ny;
    ss_nx >> nx;
    ss_ny >> ny;

    char c_use_deltas;
    ss_delta >> c_use_deltas;
    if (c_use_deltas != 'y' && c_use_deltas != 'n') {
        std::cout << "Please specify y/n for 4th argument" << std::endl;
        return -1;
    }
    bool use_deltas = c_use_deltas == 'y';

    // Load an image.
    cv::Mat image;
    image = cv::imread(argv[1], cv::IMREAD_COLOR);

    if( !image.data ) {
        std::cout <<  "Could not open or find the image" << std::endl;
        return -1;
    }

    std::cout << "Fitting " << image.rows << "x" << image.cols << " image" << std::endl;

    auto pcolor_num = image.rows * image.cols; 
    auto pcolor_sz = pcolor_num * sizeof(float) * 3;
    
    auto tcolor_num = nx * ny * 2; 
    auto tcolor_sz = tcolor_num * sizeof(float) * 3;

    auto vertices_num = (nx + 1) * (ny + 1);
    auto vertices_sz = vertices_num * sizeof(float) * 2;

    auto indices_num = nx * ny * 2;
    auto indices_sz = indices_num * sizeof(int) * 3;

    auto image_pcs = image.rows * image.cols * 3;
    auto image_sz = image_pcs * sizeof(float);

    Image* tri_image;
    LinearFragment* colors;
    DLinearFragment* d_colors;
    TriMesh* mesh;
    DTriMesh* d_mesh;

    cudaMallocManaged(&tri_image, sizeof(Image));
    cudaMallocManaged(&mesh, sizeof(TriMesh));

    build_initial_triangles(mesh, nx, ny, image.rows, image.cols);
    build_d_mesh(mesh, &d_mesh);

    std::cout << "Build meshes" << std::endl;
    // Const fragments.
    build_linear_colors(mesh, &colors);
    build_d_linear_colors(mesh, &d_colors);

    std::cout << "Build colors" << std::endl;

    float* triangle_image;
    char* triangle_bimage = (char*) malloc(image_pcs * 1);
    cudaMallocManaged(&triangle_image,    image_sz);

    float* loss_image;
    char* loss_bimage = (char*) malloc(image.rows * image.cols * 1);
    cudaMallocManaged(&loss_image, sizeof(float) * image.rows * image.cols);

    int max_jobs = image.rows * image.cols * 10 * sizeof(int);
    int* tids;
    int* pids;

    cudaMallocManaged(&tids,     max_jobs * sizeof(int));
    cudaMallocManaged(&pids,     max_jobs * sizeof(int));

    std::cout << type2str(image.type()) << std::endl;

    tri_image->rows = image.rows;
    tri_image->cols = image.cols;
    cudaMallocManaged(&(tri_image->colors), sizeof(Color) * image.rows * image.cols);
    // Load image data.
    for(int i = 0; i < image.rows; i++)
        for(int j = 0; j < image.cols; j++){
            cv::Vec3b v = image.at<cv::Vec3b>(i, j);
            tri_image->colors[(image.cols * i + j)].r = ((float)v[0]) / 255.0;//*(image.data + idx + 0);
            tri_image->colors[(image.cols * i + j)].g = ((float)v[1]) / 255.0;//*(image.data + idx + 1);
            tri_image->colors[(image.cols * i + j)].b = ((float)v[2]) / 255.0;//*(image.data + idx + 2);
        }

    //x = (float*) malloc(N*sizeof(float));
    //y = (float*) malloc(N*sizeof(float));

    int num_jobs = 0;

    std::stringstream loss_string;
    for (int iter = 0; iter < 150; iter ++){
        printf("Iteration %d", iter);

        // Zero buffers.
        // set_zero<<<((tcolor_num * 3) / 256), 256>>>(d_tcolors);
        // set_zero<<<((vertices_num * 2) / 256), 256>>>(d_vertices);
        set_zero<DLinearFragment><<<((mesh->num_triangles) / 256 + 1), 256>>>(d_colors, mesh->num_triangles);
        set_zero<DVertex><<<((mesh->num_vertices) / 256 + 1), 256>>>(d_mesh->d_vertices, mesh->num_triangles);
        set_zero<float><<<(image_pcs / 256) + 1, 256>>>(triangle_image, image_pcs);
        set_zero<float><<<(image.rows * image.cols / 256) + 1, 256>>>(loss_image, image.rows * image.cols);

        num_jobs = generate_jobs(image.rows, image.cols, mesh, tids, pids);
        printf("jobs: %d\n", num_jobs);
        assert(num_jobs <= max_jobs);

        cudaDeviceSynchronize();

        // Compute derivatives.
        if (use_deltas) {
            linear_deriv_kernel<<<(num_jobs / 256) + 1, 256>>>(
                tids,
                pids,
                num_jobs,
                tri_image,
                mesh,
                d_mesh,
                colors,
                d_colors);
        } else {
            linear_deriv_kernel_nodelta<<<(num_jobs / 256) + 1, 256>>>(
                tids,
                pids,
                num_jobs,
                tri_image,
                mesh,
                d_mesh,
                colors,
                d_colors);
        }

        cudaDeviceSynchronize();

        linear_loss_kernel<<<(num_jobs / 256) + 1, 256>>>(
            tids,
            pids,
            num_jobs,
            tri_image,
            mesh,
            colors,
            loss_image);

        cudaDeviceSynchronize();
        // TODO: temp disable regularization
        // compute_triangle_regularization(mesh, d_mesh);
        // Update values.
        /*update_values<<< (nx * ny) / 256 + 1, 256 >>>(
            nx, ny, vertices, tcolors, d_vertices, d_tcolors, ALPHA
        );*/
        compute_triangle_regularization(mesh, d_mesh, 30);
        float avg_total_pixel_area = image.rows * image.cols / (nx * ny);
        float avg_triangle_surface_area = image.rows * image.cols / (sqrt(nx * ny));

        update_vertices<<< (mesh->num_vertices) / 256 + 1, 256 >>>(
            mesh, d_mesh, ALPHA_VERTEX / avg_triangle_surface_area
        );

        update_linear_colors<<< (mesh->num_triangles) / 256 + 1, 256 >>>(
            mesh->num_triangles, colors, d_colors, ALPHA_COLOR / avg_total_pixel_area
        );

        cudaDeviceSynchronize();

        // Render triangles to image.
        linear_integral_kernel<<<(num_jobs / 256) + 1, 256>>>(
                    tids,
                    pids,
                    num_jobs,
                    mesh,
                    colors,
                    triangle_image,
                    image.rows, image.cols);

        cudaDeviceSynchronize();

        for(int idx = 0; idx < image_pcs; idx ++){
            int _val = (int)(triangle_image[idx] * 256);
            triangle_bimage[idx] = (char) ((_val < 0) ? 0 : (_val > 255 ? 255 : _val));
        }

        float total_loss = 0.f;
        for(int idx = 0; idx < image.rows * image.cols; idx ++){
            int _val = (int)(loss_image[idx] * 256);
            total_loss += loss_image[idx];
            loss_bimage[idx] = (char) ((_val < 0) ? 0 : (_val > 255 ? 255 : _val));
        }
        
        loss_string << total_loss << std::endl;
        std::cout << "Loss: " << total_loss << std::endl;

        std::stringstream ss;
        ss << "iter-" << iter << ".png";
        cv::imwrite(ss.str(), cv::Mat(image.rows, image.cols, CV_8UC3, triangle_bimage));
        std::stringstream ss_loss;
        ss_loss << "loss-" << iter << ".png";
        cv::imwrite(ss_loss.str(), cv::Mat(image.rows, image.cols, CV_8UC1, loss_bimage));
    }

    std::ofstream outfile("out.loss");
    outfile << loss_string.str();
    outfile.close();


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
    
    cudaFree(triangle_image);
    cudaFree(tids);
    cudaFree(pids);

}