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
// Output generated from Teg
#include "tegpixel.h" 
#include "tegloss.h" 
#include "renderpixel.h"

#define ALPHA_COLOR 0.7
#define ALPHA_VERTEX 3000

__global__
void pt_loss_derivative(int* tids,
                        int* pids,
                        int num_jobs,
                        Image* image,
                        TriMesh* mesh,
                        DTriMesh* d_mesh,
                        ConstFragment* colors,
                        DConstFragment* d_colors)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= num_jobs)
        return;

    auto tri_id = tids[idx];
    auto pixel_id = pids[idx];

    Vertex* a = mesh->tv0(tri_id);
    Vertex* b = mesh->tv1(tri_id);
    Vertex* c = mesh->tv2(tri_id);

    Color pixel = image->colors[pixel_id];
    Color tricolor = colors[tri_id].c;

    int h = image->cols;
    // Run generated teg function.
    auto outvals = tegpixel(
        a->x,a->y,
        b->x,b->y,
        c->x,c->y,

        floorf(pixel_id / h),
        floorf(pixel_id / h) + 1,
        (float)(pixel_id % h),
        (float)(pixel_id % h + 1),

        pixel.r,
        pixel.g,
        pixel.b,

        tricolor.r,
        tricolor.g,
        tricolor.b
    );

    DVertex* d_a = d_mesh->tv0(tri_id);
    DVertex* d_b = d_mesh->tv1(tri_id);
    DVertex* d_c = d_mesh->tv2(tri_id);

    DConstFragment *d_pcolor = d_colors + tri_id;

    // Accumulate derivatives.
    // TODO: There needs to be an easier way to accumulate derivatives..
    atomicAdd(&d_a->x, outvals[0]);
    atomicAdd(&d_b->x, outvals[1]);
    atomicAdd(&d_c->x, outvals[2]);

    atomicAdd(&d_a->y, outvals[3]);
    atomicAdd(&d_b->y, outvals[4]);
    atomicAdd(&d_c->y, outvals[5]);

    atomicAdd(&d_pcolor->d_c.r, outvals[13]);
    atomicAdd(&d_pcolor->d_c.g, outvals[14]);
    atomicAdd(&d_pcolor->d_c.b, outvals[15]);

}

__global__
void pt_loss(int* tids,
            int* pids,
            int num_jobs,
            Image* image,
            TriMesh* mesh,
            ConstFragment* colors,
            float* loss_image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= num_jobs)
        return;

    auto tri_id = tids[idx];
    auto pixel_id = pids[idx];

    Vertex* a = mesh->tv0(tri_id);
    Vertex* b = mesh->tv1(tri_id);
    Vertex* c = mesh->tv2(tri_id);

    Color pixel = image->colors[pixel_id];
    Color tricolor = colors[tri_id].c;

    int h = image->cols;
    // Run generated teg function.
    auto outval = pixel_loss(
        a->x,a->y,
        b->x,b->y,
        c->x,c->y,

        floorf(pixel_id / h),
        floorf(pixel_id / h) + 1,
        (float)(pixel_id % h),
        (float)(pixel_id % h + 1),

        pixel.r,
        pixel.g,
        pixel.b,

        tricolor.r,
        tricolor.g,
        tricolor.b
    );

    // Accumulate derivatives.
    // TODO: There needs to be an easier way to accumulate derivatives..
    atomicAdd(&loss_image[pixel_id], outval);

}


__global__
void render_triangles(int* tids,
                      int* pids,
                      int num_jobs,
                      TriMesh* mesh,
                      ConstFragment* colors,
                      float* image,
                      int w, int h)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_jobs) return;

    auto tri_id = tids[idx];
    auto pixel_id = pids[idx];

    Vertex* a = mesh->tv0(tri_id);
    Vertex* b = mesh->tv1(tri_id);
    Vertex* c = mesh->tv2(tri_id);

    Color tricolor = colors[tri_id].c;

    // Run generated teg function.
    auto outvals = renderpixel(
        a->x,a->y,
        b->x,b->y,
        c->x,c->y,

        floorf(pixel_id / h),
        floorf(pixel_id / h) + 1,
        (float)(pixel_id % h),
        (float)(pixel_id % h + 1),

        tricolor.r,
        tricolor.g,
        tricolor.b
    );

    // Accumulate image.
    atomicAdd(&image[pixel_id * 3 + 0], outvals[0]);
    atomicAdd(&image[pixel_id * 3 + 1], outvals[1]);
    atomicAdd(&image[pixel_id * 3 + 2], outvals[2]);
}


__global__
void update_const_colors(int num_colors,
                   ConstFragment* colors,
                   DConstFragment* d_colors,
                   float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > num_colors)
        return;

    Color* c = &colors[idx].c;
    DColor* d_c = &d_colors[idx].d_c;
    c->r = c->r - alpha * d_c->r;
    c->g = c->g - alpha * d_c->g;
    c->b = c->b - alpha * d_c->b;
}

__host__
void build_const_colors(TriMesh* trimesh,
                        ConstFragment** colors) {

    cudaMallocManaged(colors, sizeof(ConstFragment) * trimesh->num_triangles);
    set_zero<ConstFragment><<<((trimesh->num_triangles) / 256) + 1, 256>>>(*colors, trimesh->num_triangles);
    cudaDeviceSynchronize();

}

__host__
void build_d_const_colors(TriMesh* trimesh,
                          DConstFragment** d_colors) {

    cudaMallocManaged(d_colors, sizeof(DConstFragment) * trimesh->num_triangles);
    set_zero<DConstFragment><<<((trimesh->num_triangles) / 256) + 1, 256>>>(*d_colors, trimesh->num_triangles);
    cudaDeviceSynchronize();

}


int main(int argc, char** argv)
{
    if (argc != 5) {
        std::cout << "Usage: ./triangulate <image_file> <tri-grid nx> <tri-grid ny> <use_deltas: y/n>" << std::endl;
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
    ConstFragment* colors;
    DConstFragment* d_colors;
    TriMesh* mesh;
    DTriMesh* d_mesh;

    cudaMallocManaged(&tri_image, sizeof(Image));
    cudaMallocManaged(&mesh, sizeof(TriMesh));

    build_initial_triangles(mesh, nx, ny, image.rows, image.cols);
    build_d_mesh(mesh, &d_mesh);

    std::cout << "Build meshes" << std::endl;
    // Const fragments.
    build_const_colors(mesh, &colors);
    build_d_const_colors(mesh, &d_colors);

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

    // Bound max jobs with some constant multiple of image size.

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
        set_zero<DConstFragment><<<((mesh->num_triangles) / 256 + 1), 256>>>(d_colors, mesh->num_triangles);
        set_zero<DVertex><<<((mesh->num_vertices) / 256 + 1), 256>>>(d_mesh->d_vertices, mesh->num_triangles);
        set_zero<float><<<(image_pcs / 256) + 1, 256>>>(triangle_image, image_pcs);
        set_zero<float><<<(image.rows * image.cols / 256) + 1, 256>>>(loss_image, image.rows * image.cols);

        num_jobs = generate_jobs(image.rows, image.cols, mesh, tids, pids);
        printf("jobs: %d\n", num_jobs);
        assert(num_jobs <= max_jobs);

        cudaDeviceSynchronize();

        // Compute derivatives.
        pt_loss_derivative<<<(num_jobs / 256) + 1, 256>>>(
            tids,
            pids,
            num_jobs,
            tri_image,
            mesh,
            d_mesh,
            colors,
            d_colors);

        cudaDeviceSynchronize();

        pt_loss<<<(num_jobs / 256) + 1, 256>>>(
            tids,
            pids,
            num_jobs,
            tri_image,
            mesh,
            colors,
            loss_image);

        compute_triangle_regularization(mesh, d_mesh, 2);
        // Update values.
        /*update_values<<< (nx * ny) / 256 + 1, 256 >>>(
            nx, ny, vertices, tcolors, d_vertices, d_tcolors, ALPHA
        );*/

        float avg_total_pixel_area = image.rows * image.cols / (nx * ny);
        float avg_triangle_surface_area = image.rows * image.cols / (sqrt(nx * ny));
        update_vertices<<< (mesh->num_vertices) / 256 + 1, 256 >>>(
            mesh, d_mesh, (use_deltas ? ALPHA_VERTEX / avg_triangle_surface_area : 0)
        );

        update_const_colors<<< (mesh->num_triangles) / 256 + 1, 256 >>>(
            mesh->num_triangles, colors, d_colors, ALPHA_COLOR / avg_total_pixel_area
        );

        cudaDeviceSynchronize();

        // Render triangles to image.
        render_triangles<<<(num_jobs / 256) + 1, 256>>>(
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

        std::cout << "Loss: " << total_loss << std::endl;
        loss_string << total_loss << std::endl;

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

    cudaFree(triangle_image);
    cudaFree(tids);
    cudaFree(pids);

}