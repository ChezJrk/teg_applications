#include "types.cuh"

#include "teg_perlin_double_thresholded.h"
#include "teg_perlin_double_threshold_fwdderiv.h"
#include "teg_perlin_noise.h"

#include <stdlib.h>



template<typename T>
__global__
void set_zero(T* data, int max_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < max_id)
        data[idx] = 0;
}

template<typename T>
__host__
void fast_zero(T* data, int size) {
    set_zero<T><<<(size/256) + 1, 256>>>(data, size);
    cudaDeviceSynchronize();
}


__global__
void update_grid_positions(Grid* grid,
                           DGrid* d_grid,
                           float alpha)
{
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= (grid->rows + 1) || y >= (grid->cols + 1))
        return;

    Vec2D* v = &grid->vecs[y * (grid->rows + 1) + x];
    DVec2D* d_v = &d_grid->d_vecs[y * (grid->rows + 1) + x];
    v->x = v->x - alpha * d_v->d_x;
    v->y = v->y - alpha * d_v->d_y;

    v->x = v->x / (v->x * v->x + v->y * v->y);
    v->y = v->y / (v->x * v->x + v->y * v->y);
}


__host__
void build_grid(int nx, int ny, Grid** grid, int seed) {

    cudaMallocManaged(grid, sizeof(Grid));
    cudaMallocManaged(&(*grid)->vecs, sizeof(Vec2D) * (nx + 1) * (ny + 1));
    (*grid)->rows = nx;
    (*grid)->cols = ny;

    srand(seed);
    // Randomize everything
    for(int x = 0; x < nx + 1; x++){
        for(int y = 0; y < ny + 1; y++){
            float theta = (static_cast<float>(rand())/RAND_MAX) * 2 * 3.1415f;
            //float theta = 0;
            (*grid)->vecs[y * (nx + 1) + x] = Vec2D{cos(theta), sin(theta)};
        }
    }

}

__host__
void build_image(Image** image, int rx, int ry) {

    cudaMallocManaged(image, sizeof(Image));
    cudaMallocManaged(&(*image)->colors, sizeof(Color) * rx * ry);
    (*image)->rows = rx;
    (*image)->cols = ry;
    
}

__host__
void build_d_grid(Grid* grid,
                  DGrid** d_grid) {

    cudaMallocManaged(d_grid, sizeof(DGrid));
    (*d_grid)->rows = grid->rows;
    (*d_grid)->cols = grid->cols;

    cudaMallocManaged(&((*d_grid)->d_vecs), sizeof(DVec2D) * (grid->rows + 1) * (grid->cols + 1));
    set_zero<DVec2D><<<((grid->rows + 1) * (grid->cols + 1) / 256) + 1, 256>>>(((*d_grid)->d_vecs),
                        (grid->rows + 1) * (grid->cols + 1));
}


__global__
void perlin_double_threshold_kernel(int* qids,
                            int* pids,
                            int num_jobs,
                            Grid* grid,
                            Color* pcolor,
                            Color* ncolor,
                            Color* mcolor,
                            float* threshold,
                            float* threshold2,
                            float* image,
                            int w, int h)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_jobs) return;

    auto quad_id = qids[idx];
    auto pixel_id = pids[idx];

    int y_low = static_cast<int>(floorf(quad_id / grid->rows));
    int y_high = static_cast<int>(floorf(quad_id / grid->rows) + 1);
    int x_low = quad_id % grid->rows;
    int x_high = ((quad_id % grid->rows) + 1);

    Vec2D vec11 = grid->vecs[y_high * (grid->rows + 1) + x_high];
    Vec2D vec10 = grid->vecs[y_low * (grid->rows + 1) + x_high];
    Vec2D vec01 = grid->vecs[y_high * (grid->rows + 1) + x_low];
    Vec2D vec00 = grid->vecs[y_low * (grid->rows + 1) + x_low];

    // printf("Sample %f, %f\n", vec11.x, vec11.y);
    // Assumes grid dimensions are always smaller than image dimensions.
    int scale_y = h / grid->cols;
    int scale_x = w / grid->rows;

    // Run generated teg function.
    auto outvals = teg_perlin_double_thresholded(
        vec00.x, vec00.y,
        vec10.x, vec10.y,
        vec01.x, vec01.y,
        vec11.x, vec11.y,

        x_low * scale_x, x_high * scale_x,
        y_low * scale_y, y_high * scale_y,

        (float)(pixel_id % w),
        (float)(pixel_id % w + 1),
        floorf(pixel_id / w),
        floorf(pixel_id / w) + 1,

        pcolor->r, pcolor->g, pcolor->b,
        ncolor->r, ncolor->g, ncolor->b,
        mcolor->r, mcolor->g, mcolor->b,

        *threshold,
        *threshold2
    );

    // printf("%d: %f, %f, %f\n", pixel_id, outvals[0], outvals[1], outvals[2]);

    // Accumulate image.
    atomicAdd(&image[pixel_id * 3 + 0], outvals[0]);
    atomicAdd(&image[pixel_id * 3 + 1], outvals[1]);
    atomicAdd(&image[pixel_id * 3 + 2], outvals[2]);
}


__global__
void perlin_double_threshold_deriv(int* qids,
                            int* pids,
                            int num_jobs,
                            Image* target_image,
                            Grid* grid,
                            DGrid* d_grid,
                            Color* pcolor,
                            Color* ncolor,
                            Color* mcolor,
                            DColor* d_pcolor,
                            DColor* d_ncolor,
                            DColor* d_mcolor,
                            float* threshold,
                            float* threshold2,
                            float* d_threshold,
                            float* d_threshold2,
                            int w, int h)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_jobs) return;

    auto quad_id = qids[idx];
    auto pixel_id = pids[idx];

    int y_low = static_cast<int>(floorf(quad_id / grid->rows));
    int y_high = static_cast<int>(floorf(quad_id / grid->rows) + 1);
    int x_low = quad_id % grid->rows;
    int x_high = ((quad_id % grid->rows) + 1);

    Vec2D vec11 = grid->vecs[y_high * (grid->rows + 1) + x_high];
    Vec2D vec10 = grid->vecs[y_low * (grid->rows + 1) + x_high];
    Vec2D vec01 = grid->vecs[y_high * (grid->rows + 1) + x_low];
    Vec2D vec00 = grid->vecs[y_low * (grid->rows + 1) + x_low];

    // printf("Sample %f, %f\n", vec11.x, vec11.y);
    // Assumes grid dimensions are always smaller than image dimensions.
    int scale_y = h / grid->cols;
    int scale_x = w / grid->rows;

    // Run generated teg function.
    auto outvals = teg_perlin_double_threshold_fwdderiv(
        vec00.x, vec00.y,
        vec10.x, vec10.y,
        vec01.x, vec01.y,
        vec11.x, vec11.y,

        x_low * scale_x, x_high * scale_x,
        y_low * scale_y, y_high * scale_y,

        (float)(pixel_id % w),
        (float)(pixel_id % w + 1),
        floorf(pixel_id / w),
        floorf(pixel_id / w) + 1,

        pcolor->r, pcolor->g, pcolor->b,
        ncolor->r, ncolor->g, ncolor->b,
        mcolor->r, mcolor->g, mcolor->b,

        target_image->colors[pixel_id].r,
        target_image->colors[pixel_id].g,
        target_image->colors[pixel_id].b,

        *threshold,
        *threshold2
    );

    // printf("%d: %f, %f, %f\n", pixel_id, outvals[0], outvals[1], outvals[2]);

    // Accumulate image.
    // atomicAdd(&image[pixel_id * 3 + 0], outval);
    // atomicAdd(&image[pixel_id * 3 + 1], outval);
    // atomicAdd(&image[pixel_id * 3 + 2], outval);

    atomicAdd(&d_grid->d_vecs[y_low * (grid->rows + 1) + x_low].d_x, outvals[0]);
    atomicAdd(&d_grid->d_vecs[y_low * (grid->rows + 1) + x_low].d_y, outvals[1]);
    atomicAdd(&d_grid->d_vecs[y_low * (grid->rows + 1) + x_high].d_x, outvals[2]);
    atomicAdd(&d_grid->d_vecs[y_low * (grid->rows + 1) + x_high].d_y, outvals[3]);
    atomicAdd(&d_grid->d_vecs[y_high * (grid->rows + 1) + x_low].d_x, outvals[4]);
    atomicAdd(&d_grid->d_vecs[y_high * (grid->rows + 1) + x_low].d_y, outvals[5]);
    atomicAdd(&d_grid->d_vecs[y_high * (grid->rows + 1) + x_high].d_x, outvals[6]);
    atomicAdd(&d_grid->d_vecs[y_high * (grid->rows + 1) + x_high].d_y, outvals[7]);

    atomicAdd(&d_pcolor->r, outvals[8]);
    atomicAdd(&d_pcolor->g, outvals[9]);
    atomicAdd(&d_pcolor->b, outvals[10]);

    atomicAdd(&d_ncolor->r, outvals[11]);
    atomicAdd(&d_ncolor->g, outvals[12]);
    atomicAdd(&d_ncolor->b, outvals[13]);

    atomicAdd(&d_mcolor->r, outvals[14]);
    atomicAdd(&d_mcolor->g, outvals[15]);
    atomicAdd(&d_mcolor->b, outvals[16]);

    atomicAdd(d_threshold, outvals[17]);
    atomicAdd(d_threshold2, outvals[18]);
}

__device__
float clamp(float x, float low, float high) {
    return (x >= high) ? (high-1) : ((x < low) ? low : x);
}

/*
__global__
void linear_deriv_kernel(int* qids,
                             int* pids,
                             int num_jobs,
                             Image* image,
                             TriMesh* mesh,
                             DTriMesh* d_mesh,
                             LinearFragment* colors,
                             DLinearFragment* d_colors)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= num_jobs)
        return;

    auto tri_id = qids[idx];
    auto pixel_id = pids[idx];

    Vertex* a = mesh->tv0(tri_id);
    Vertex* b = mesh->tv1(tri_id);
    Vertex* c = mesh->tv2(tri_id);

    Color pixel = image->colors[pixel_id];
    Color tricolor0 = colors[tri_id].c0;
    Color tricolor1 = colors[tri_id].c1;
    Color tricolor2 = colors[tri_id].c2;

    int h = image->cols;
    // Run generated teg function.
    auto outvals = teg_linear_deriv(
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

        tricolor0.r,
        tricolor0.g,
        tricolor0.b,

        tricolor1.r,
        tricolor1.g,
        tricolor1.b,

        tricolor2.r,
        tricolor2.g,
        tricolor2.b
    );

    DVertex* d_a = d_mesh->tv0(tri_id);
    DVertex* d_b = d_mesh->tv1(tri_id);
    DVertex* d_c = d_mesh->tv2(tri_id);

    DLinearFragment *d_pcolor = d_colors + tri_id;

    // Accumulate derivatives.
    // TODO: There needs to be an easier way to accumulate derivatives..
    atomicAdd(&d_a->x, outvals[0]);
    atomicAdd(&d_b->x, outvals[1]);
    atomicAdd(&d_c->x, outvals[2]);

    atomicAdd(&d_a->y, outvals[3]);
    atomicAdd(&d_b->y, outvals[4]);
    atomicAdd(&d_c->y, outvals[5]);

    atomicAdd(&d_pcolor->d_c0.r, outvals[6]);
    atomicAdd(&d_pcolor->d_c0.g, outvals[7]);
    atomicAdd(&d_pcolor->d_c0.b, outvals[8]);

    atomicAdd(&d_pcolor->d_c1.r, outvals[9]);
    atomicAdd(&d_pcolor->d_c1.g, outvals[10]);
    atomicAdd(&d_pcolor->d_c1.b, outvals[11]);

    atomicAdd(&d_pcolor->d_c2.r, outvals[12]);
    atomicAdd(&d_pcolor->d_c2.g, outvals[13]);
    atomicAdd(&d_pcolor->d_c2.b, outvals[14]);

}*/