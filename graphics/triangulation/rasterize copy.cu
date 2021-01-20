#include <stdio.h>
#include "math.h"
#include <string>

// Image IO
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>

// Output generated from Teg
#include "tegpixel.h" 
#include "renderpixel.h"

// End Temporary placeholder.
#define ALPHA 0.001

__global__
void pt_loss_derivative(int w, int h, 
                        int* tids,
                        int* pids,
                        int num_jobs,
                        float *vertices, 
                        int *indices,
                        float *tcolors, float* pcolors, 
                        float *d_vertices, float *d_tcolors)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= num_jobs)
        return;

    auto tri_id = tids[idx];
    auto pixel_id = pids[idx];

    auto index1 = indices[tri_id * 3 + 0];
    auto index2 = indices[tri_id * 3 + 1];
    auto index3 = indices[tri_id * 3 + 2];

    // Run generated teg function.
    auto outvals = tegpixel(
        vertices[index1 * 2 + 0],
        vertices[index1 * 2 + 1],
        vertices[index2 * 2 + 0],
        vertices[index2 * 2 + 1],
        vertices[index3 * 2 + 0],
        vertices[index3 * 2 + 1],

        floorf(pixel_id / h),
        floorf(pixel_id / h) + 1,
        (float)(pixel_id % h),
        (float)(pixel_id % h + 1),

        pcolors[pixel_id * 3 + 0],
        pcolors[pixel_id * 3 + 1],
        pcolors[pixel_id * 3 + 2],

        tcolors[tri_id * 3 + 0],
        tcolors[tri_id * 3 + 1],
        tcolors[tri_id * 3 + 2]
    );

    /*auto temp_outvals = tegpixel(
        0, -2,
        0, 2,
        1, 0,
        -1, 1,
        -1, 1,
        1, 1, 1,
        0.5, 0.5, 0.5
    );
    if (idx == 0){
        printf("%f, %f, %f, %f, %f, %f\n", 
            temp_outvals.o[0], temp_outvals.o[1], temp_outvals.o[2],
            temp_outvals.o[3], temp_outvals.o[4], temp_outvals.o[5]);
    }*/
    /*
    if (index3 == 366 && outvals.o[0] != 0.f){
        printf("\
                Hello from block %d, thread %d\n\
                Tri_id %d, Pix_id %d\n\
                Pix-color %f, %f, %f\n\
                T-colors %f, %f, %f\n\
                Out-vals %f, %f, %f\n\
                0: %f, %f\n\
                1: %f, %f\n\
                2: %f, %f\n\
                x: %f, %f\n\
                y: %f, %f\n\
                idxs: %d, %d, %d\n", 
                    blockIdx.x, threadIdx.x,
                    tri_id, pixel_id,
                    pcolors[pixel_id * 3 + 0], pcolors[pixel_id * 3 + 1], pcolors[pixel_id * 3 + 2],
                    tcolors[tri_id * 3 + 0], tcolors[tri_id * 3 + 1], tcolors[tri_id * 3 + 2],
                    outvals.o[0], outvals.o[1], outvals.o[2],
                    vertices[index1 * 2 + 0], vertices[index1 * 2 + 1],
                    vertices[index2 * 2 + 0], vertices[index2 * 2 + 1],
                    vertices[index3 * 2 + 0], vertices[index3 * 2 + 1],
                    floorf(pixel_id / h),
                    floorf(pixel_id / h) + 1,
                    (float)(pixel_id % h),
                    (float)(pixel_id % h + 1),
                    index1, index2, index3);
    }*/

    // Accumulate derivatives.
    // TODO: There needs to be an easier way to accumulate derivatives..
    atomicAdd(&d_vertices[index1 * 2 + 0], outvals[0]);
    atomicAdd(&d_vertices[index2 * 2 + 0], outvals[1]);
    atomicAdd(&d_vertices[index3 * 2 + 0], outvals[2]);

    atomicAdd(&d_vertices[index1 * 2 + 1], outvals[3]);
    atomicAdd(&d_vertices[index2 * 2 + 1], outvals[4]);
    atomicAdd(&d_vertices[index3 * 2 + 1], outvals[5]);

    //if (index3 == 366) {
    //    printf("pt_loss_derivative for %d: %f\n", index3, d_vertices[index3 * 2 + 0]);
    //}

    atomicAdd(&d_tcolors[tri_id * 3 + 0], outvals[13]);
    atomicAdd(&d_tcolors[tri_id * 3 + 1], outvals[14]);
    atomicAdd(&d_tcolors[tri_id * 3 + 2], outvals[15]);
    
}

__global__
void render_triangles(int w, int h, 
                        int* tids,
                        int* pids,
                        int num_jobs,
                        float *vertices, 
                        int *indices,
                        float *tcolors,
                        float *image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_jobs) return;

    auto tri_id = tids[idx];
    auto pixel_id = pids[idx];

    auto index1 = indices[tri_id * 3 + 0];
    auto index2 = indices[tri_id * 3 + 1];
    auto index3 = indices[tri_id * 3 + 2];

    // Run generated teg function.
    auto outvals = renderpixel(
        vertices[index1 * 2 + 0],
        vertices[index1 * 2 + 1],
        vertices[index2 * 2 + 0],
        vertices[index2 * 2 + 1],
        vertices[index3 * 2 + 0],
        vertices[index3 * 2 + 1],

        floorf(pixel_id / h),
        floorf(pixel_id / h) + 1,
        (float)(pixel_id % h),
        (float)(pixel_id % h + 1),

        tcolors[tri_id * 3 + 0],
        tcolors[tri_id * 3 + 1],
        tcolors[tri_id * 3 + 2]
    );

    // Accumulate image.
    atomicAdd(&image[pixel_id * 3 + 0], outvals[0]);
    atomicAdd(&image[pixel_id * 3 + 1], outvals[1]);
    atomicAdd(&image[pixel_id * 3 + 2], outvals[2]);
}

__global__
void update_values(int nx, int ny,
                     float *vertices, float *tcolors,
                     float *d_vertices, float *d_tcolors,
                     float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int x = idx / (ny + 1);
    int y = idx % (ny + 1);
    if (x > 0 && y > 0 && x < nx && y < ny){
        vertices[idx * 2 + 0] = vertices[idx * 2 + 0] - alpha * 100 * d_vertices[idx * 2 + 0];
        vertices[idx * 2 + 1] = vertices[idx * 2 + 1] - alpha * 100 * d_vertices[idx * 2 + 1];
    }

    x = idx / (ny);
    y = idx % (ny);
    if (x >= 0 && y >= 0 && x < nx && y < ny){
        tcolors[idx * 6 + 0]  = tcolors[idx * 6 + 0] - alpha * d_tcolors[idx * 6 + 0];
        tcolors[idx * 6 + 1]  = tcolors[idx * 6 + 1] - alpha * d_tcolors[idx * 6 + 1];
        tcolors[idx * 6 + 2]  = tcolors[idx * 6 + 2] - alpha * d_tcolors[idx * 6 + 2];

        tcolors[idx * 6 + 3]  = tcolors[idx * 6 + 3] - alpha * d_tcolors[idx * 6 + 3];
        tcolors[idx * 6 + 4]  = tcolors[idx * 6 + 4] - alpha * d_tcolors[idx * 6 + 4];
        tcolors[idx * 6 + 5]  = tcolors[idx * 6 + 5] - alpha * d_tcolors[idx * 6 + 5];
    }
}

__global__ 
void set_zero(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = 0.f;
}

__host__
void build_initial_triangles(float* vertices, int* indices, 
                             float* tcolors,
                             int nx, int ny, 
                             int image_width, int image_height) {

    float tri_width  = (float)(image_width)  / nx;
    float tri_height = (float)(image_height) / ny;

    for(int i = 0; i < nx + 1; i++) {
        for(int j = 0; j < ny + 1; j++) {
            vertices[(i * (ny + 1) + j) * 2 + 0] = tri_width  * i;
            vertices[(i * (ny + 1) + j) * 2 + 1] = tri_height * j;
        }
    }

    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
            indices[(i * ny + j) * 6 + 0] = ((i + 0) * (ny + 1) + j + 0);
            indices[(i * ny + j) * 6 + 1] = ((i + 0) * (ny + 1) + j + 1);
            indices[(i * ny + j) * 6 + 2] = ((i + 1) * (ny + 1) + j + 1);

            indices[(i * ny + j) * 6 + 3] = ((i + 0) * (ny + 1) + j + 0);
            indices[(i * ny + j) * 6 + 4] = ((i + 1) * (ny + 1) + j + 1);
            indices[(i * ny + j) * 6 + 5] = ((i + 1) * (ny + 1) + j + 0);
        }
    }

    for(int i = 0; i < nx; i++)
        for(int j = 0; j < ny; j++) {
            tcolors[((i * ny) + j) * 6 + 0] = 0.f;
            tcolors[((i * ny) + j) * 6 + 1] = 0.f;
            tcolors[((i * ny) + j) * 6 + 2] = 0.f;

            tcolors[((i * ny) + j) * 6 + 3] = 0.f;
            tcolors[((i * ny) + j) * 6 + 4] = 0.f;
            tcolors[((i * ny) + j) * 6 + 5] = 0.f;
        }
}

int generate_jobs( int image_width, int image_height, 
                    int nx, int ny, 
                    float* vertices, int* indices, 
                    int* tids, int* pids ) {
    int job_count = 0;
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for(int t = 0; t < 2; t++) {
                int idx = (i * ny + j) * 2 + t;

                int i0 = indices[idx * 3 + 0];
                int i1 = indices[idx * 3 + 1];
                int i2 = indices[idx * 3 + 2];

                float vx0 = vertices[i0 * 2 + 0];
                float vy0 = vertices[i0 * 2 + 1];

                float vx1 = vertices[i1 * 2 + 0];
                float vy1 = vertices[i1 * 2 + 1];

                float vx2 = vertices[i2 * 2 + 0];
                float vy2 = vertices[i2 * 2 + 1];

                int max_x = (int)(std::ceil( std::max(vx0, std::max(vx1, vx2)))) + 1;
                int max_y = (int)(std::ceil( std::max(vy0, std::max(vy1, vy2)))) + 1;
                int min_x = (int)(std::floor( std::min(vx0, std::min(vx1, vx2)))) - 1;
                int min_y = (int)(std::floor( std::min(vy0, std::min(vy1, vy2)))) - 1;

                if (min_x < 0) min_x = 0;
                if (min_y < 0) min_y = 0;
                if (max_x >= image_width) max_x = image_width - 1;
                if (max_y >= image_height) max_y = image_height - 1;

                for (int tx = min_x; tx < max_x; tx++) {
                    for (int ty = min_y; ty < max_y; ty++) {
                        tids[job_count] = idx;
                        pids[job_count] = tx * image_height + ty;
                        //if(job_count % 100000 == 0) std::cout << job_count << " " << idx << " " << i0 << " " << i1 << " " << i2 << std::endl;
                        job_count ++;
                    }
                }
            }
        }
    }

    return job_count;
}

float sgn(float x){
    return (x > 0) ? 1: -1;
}

int compute_triangle_regularization( int image_width, int image_height, 
    int nx, int ny, 
    float* vertices, int* indices, 
    float* d_vertices) {
    int job_count = 0;
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for(int t = 0; t < 2; t++) {
                int idx = (i * ny + j) * 2 + t;

                int i0 = indices[idx * 3 + 0];
                int i1 = indices[idx * 3 + 1];
                int i2 = indices[idx * 3 + 2];

                float vx0 = vertices[i0 * 2 + 0];
                float vy0 = vertices[i0 * 2 + 1];

                float vx1 = vertices[i1 * 2 + 0];
                float vy1 = vertices[i1 * 2 + 1];

                float vx2 = vertices[i2 * 2 + 0];
                float vy2 = vertices[i2 * 2 + 1];
                
                float cx = (vx0 + vx1 + vx2) / 3;
                float cy = (vy0 + vy1 + vy2) / 3;

                // Cross-product for area.
                float area = ((vx0 - vx1)*(vy0 - vy2) - (vx0 - vx2)*(vy0 - vy1)) / 8.f;
                /*
                d_vertices[i0 * 2 + 0] += (-1.0/area) * sgn(vx0 - cx);
                d_vertices[i0 * 2 + 1] += (-1.0/area) * sgn(vy0 - cy);

                d_vertices[i1 * 2 + 0] += (-1.0/area) * sgn(vx1 - cx);
                d_vertices[i1 * 2 + 1] += (-1.0/area) * sgn(vy1 - cy);

                d_vertices[i2 * 2 + 0] += (-1.0/area) * sgn(vx2 - cx);
                d_vertices[i2 * 2 + 1] += (-1.0/area) * sgn(vy2 - cy);
                */
                d_vertices[i0 * 2 + 0] += (-1.0/area) * (vy1 - vy2);
                d_vertices[i0 * 2 + 1] += (-1.0/area) * (vx2 - vx1);

                d_vertices[i1 * 2 + 0] += (-1.0/area) * (vy2 - vy0);
                d_vertices[i1 * 2 + 1] += (-1.0/area) * (vx0 - vx2);

                d_vertices[i2 * 2 + 0] += (-1.0/area) * (vy0 - vy1);
                d_vertices[i2 * 2 + 1] += (-1.0/area) * (vx1 - vx0);

            }
        }
    }

    return job_count;
}

std::string type2str(int type) {
    std::string r;
  
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
  
    switch ( depth ) {
      case CV_8U:  r = "8U"; break;
      case CV_8S:  r = "8S"; break;
      case CV_16U: r = "16U"; break;
      case CV_16S: r = "16S"; break;
      case CV_32S: r = "32S"; break;
      case CV_32F: r = "32F"; break;
      case CV_64F: r = "64F"; break;
      default:     r = "User"; break;
    }
  
    r += "C";
    r += (chans+'0');
  
    return r;
}

int main(int argc, char** argv)
{
    int nx = 30;
    int ny = 30;

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

    /*
    float* pcolors  =  (float*) malloc(pcolor_sz);
    float* tcolors  =  (float*) malloc(tcolor_sz);
    float* vertices =  (float*) malloc(vertices_sz);
    int*   indices  =  (int*)   malloc(indices_sz);

    float* d_vertices = (float*) malloc(vertices_sz);
    float* d_tcolors  = (float*) malloc(tcolor_sz);

    float* triangle_image = (float*) malloc(image_sz);
    char* triangle_bimage = (char*) malloc(image_pcs * 1);
    */
    float* pcolors;
    float* tcolors;
    float* vertices;
    int*   indices;

    float* d_vertices;
    float* d_tcolors;

    float* triangle_image;
    char* triangle_bimage = (char*) malloc(image_pcs * 1);

    int max_jobs = image.rows * image.cols * 10 * sizeof(int);
    int* tids;
    int* pids;

    cudaMallocManaged(&pcolors,  pcolor_sz); 
    cudaMallocManaged(&tcolors,  tcolor_sz);
    cudaMallocManaged(&vertices, vertices_sz);
    cudaMallocManaged(&indices,  indices_sz);
    cudaMallocManaged(&tids,     max_jobs * sizeof(int));
    cudaMallocManaged(&pids,     max_jobs * sizeof(int));
    cudaMallocManaged(&triangle_image,    image_sz);

    cudaMallocManaged(&d_tcolors,  tcolor_sz);
    cudaMallocManaged(&d_vertices, vertices_sz);

    build_initial_triangles(vertices, indices, tcolors, nx, ny, image.rows, image.cols);

    cudaDeviceSynchronize();

    //int num_jobs = generate_jobs(image.rows, image.cols, nx, ny, vertices, indices, tids, pids);
    //std::cout<< "Generated " << num_jobs << " jobs. " << "(Max:" << max_jobs << ")"  << std::endl;

    // Bound max jobs with some constant multiple of image size.

    std::cout << type2str(image.type()) << std::endl;

    // Load image data.
    for(int i = 0; i < image.rows; i++)
        for(int j = 0; j < image.cols; j++){
            //int idx = (image.cols * i + j) * 3;
            cv::Vec3b v = image.at<cv::Vec3b>(i, j);
            pcolors[(image.cols * i + j) * 3 + 0] = ((float)v[0]) / 255.0;//*(image.data + idx + 0);
            pcolors[(image.cols * i + j) * 3 + 1] = ((float)v[1]) / 255.0;//*(image.data + idx + 1);
            pcolors[(image.cols * i + j) * 3 + 2] = ((float)v[2]) / 255.0;//*(image.data + idx + 2);
        }

    //x = (float*) malloc(N*sizeof(float));
    //y = (float*) malloc(N*sizeof(float));



    int num_jobs = 0;

    for (int iter = 0; iter < 150; iter ++){
        printf("Iteration %d", iter);

        // Zero buffers.
        set_zero<<<((tcolor_num * 3) / 256), 256>>>(d_tcolors);
        set_zero<<<((vertices_num * 2) / 256), 256>>>(d_vertices);
        set_zero<<<(image_pcs / 256), 256>>>(triangle_image);

        num_jobs = generate_jobs(image.rows, image.cols, nx, ny, vertices, indices, tids, pids);
        printf("jobs: %d\n", num_jobs);
        assert(num_jobs <= max_jobs);

        cudaDeviceSynchronize();

        // Compute derivatives.
        pt_loss_derivative<<<(num_jobs / 256) + 1, 256>>>(
            image.rows, image.cols,
            tids,
            pids,
            num_jobs,
            vertices,
            indices,
            tcolors, pcolors,
            d_vertices, d_tcolors);

        cudaDeviceSynchronize();
        compute_triangle_regularization(image.rows, image.cols, nx, ny, vertices, indices, d_vertices);
        // Update values.
        update_values<<< (nx * ny) / 256 + 1, 256 >>>(
            nx, ny, vertices, tcolors, d_vertices, d_tcolors, ALPHA
        );

        cudaDeviceSynchronize();

        // Render triangles to image.
        render_triangles<<<(num_jobs / 256) + 1, 256>>>(
            image.rows, image.cols, 
            tids,
            pids,
            num_jobs,
            vertices,
            indices,
            tcolors,
            triangle_image);

        cudaDeviceSynchronize();

        for(int idx = 0; idx < image_pcs; idx ++){
            int _val = (int)(triangle_image[idx] * 256);
            triangle_bimage[idx] = (char) ((_val < 0) ? 0 : (_val > 255 ? 255 : _val));
        }

        std::stringstream ss;
        ss << "iter-" << iter << ".png";
        cv::imwrite(ss.str(), cv::Mat(image.rows, image.cols, CV_8UC3, triangle_bimage));
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


    cudaFree(pcolors);
    cudaFree(tcolors);
    cudaFree(vertices);
    cudaFree(indices);
    cudaFree(tids);
    cudaFree(pids);

}