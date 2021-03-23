#include "triangle.cuh"

#include "teg_linear_integral.h"
#include "teg_linear_deriv.h"
#include "teg_linear_loss.h"
#include "teg_linear_deriv_nodelta.h"
#include "teg_linear_bilinear_deriv.h"

__global__
void update_linear_colors(int num_colors,
                   LinearFragment* colors,
                   DLinearFragment* d_colors,
                   float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > num_colors)
        return;

    Color* c0 = &colors[idx].c0;
    DColor* d_c0 = &d_colors[idx].d_c0;
    c0->r = c0->r - alpha * d_c0->r;
    c0->g = c0->g - alpha * d_c0->g;
    c0->b = c0->b - alpha * d_c0->b;

    Color* c1 = &colors[idx].c1;
    DColor* d_c1 = &d_colors[idx].d_c1;
    c1->r = c1->r - alpha * d_c1->r;
    c1->g = c1->g - alpha * d_c1->g;
    c1->b = c1->b - alpha * d_c1->b;

    Color* c2 = &colors[idx].c2;
    DColor* d_c2 = &d_colors[idx].d_c2;
    c2->r = c2->r - alpha * d_c2->r;
    c2->g = c2->g - alpha * d_c2->g;
    c2->b = c2->b - alpha * d_c2->b;
}


__host__
void build_linear_colors(TriMesh* trimesh,
                         LinearFragment** colors) {

    cudaMallocManaged(colors, sizeof(LinearFragment) * trimesh->num_triangles);
    set_zero<LinearFragment><<<((trimesh->num_triangles) / 256) + 1, 256>>>(*colors, trimesh->num_triangles);
    cudaDeviceSynchronize();

}

__host__
void build_d_linear_colors(TriMesh* trimesh,
                           DLinearFragment** d_colors) {

    cudaMallocManaged(d_colors, sizeof(DLinearFragment) * trimesh->num_triangles);
    set_zero<DLinearFragment><<<((trimesh->num_triangles) / 256) + 1, 256>>>(*d_colors, trimesh->num_triangles);
    cudaDeviceSynchronize();

}


__global__
void linear_integral_kernel(int* tids,
                      int* pids,
                      int num_jobs,
                      TriMesh* mesh,
                      LinearFragment* colors,
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

    Color tricolor0 = colors[tri_id].c0;
    Color tricolor1 = colors[tri_id].c1;
    Color tricolor2 = colors[tri_id].c2;

    // Run generated teg function.
    auto outvals = teg_linear_integral(
        a->x,a->y,
        b->x,b->y,
        c->x,c->y,

        floorf(pixel_id / h),
        floorf(pixel_id / h) + 1,
        (float)(pixel_id % h),
        (float)(pixel_id % h + 1),

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

    // Accumulate image.
    atomicAdd(&image[pixel_id * 3 + 0], outvals[0]);
    atomicAdd(&image[pixel_id * 3 + 1], outvals[1]);
    atomicAdd(&image[pixel_id * 3 + 2], outvals[2]);
}

__device__
float clamp(float x, float low, float high) {
    return (x >= high) ? (high-1) : ((x < low) ? low : x);
}

__global__
void linear_bilinear_deriv_kernel(int* tids,
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

    auto tri_id = tids[idx];
    auto pixel_id = pids[idx];

    Vertex* a = mesh->tv0(tri_id);
    Vertex* b = mesh->tv1(tri_id);
    Vertex* c = mesh->tv2(tri_id);

    int num_pixels = image->cols * image->rows;
    int h = image->cols;
    int w = image->rows;

    float x0 = floorf(pixel_id / h);
    float x1 = floorf(pixel_id / h) + 1;
    float y0 = (float)(pixel_id % h);
    float y1 = (float)(pixel_id % h + 1);

    int pix00 = (int)(roundf(clamp(x0, 0, w)) * h + roundf(clamp(y0, 0, h)));
    int pix01 = (int)(roundf(clamp(x0 + 1, 0, w)) * h + roundf(clamp(y0, 0, h)));
    int pix10 = (int)(roundf(clamp(x0, 0, w)) * h + roundf(clamp(y0 + 1, 0, h)));
    int pix11 = (int)(roundf(clamp(x0 + 1, 0, w)) * h + roundf(clamp(y0 + 1, 0, h)));

    Color pixel00 = image->colors[pix00];
    Color pixel01 = image->colors[pix01];
    Color pixel10 = image->colors[pix10];
    Color pixel11 = image->colors[pix11];

    Color tricolor0 = colors[tri_id].c0;
    Color tricolor1 = colors[tri_id].c1;
    Color tricolor2 = colors[tri_id].c2;


    // Run generated teg function.
    auto outvals = teg_linear_bilinear_deriv(
        a->x,a->y,
        b->x,b->y,
        c->x,c->y,

        x0, x1,
        y0, y1,

        pixel00.r, pixel00.g, pixel00.b,
        pixel01.r, pixel01.g, pixel01.b,
        pixel10.r, pixel10.g, pixel10.b,
        pixel11.r, pixel11.g, pixel11.b,

        tricolor0.r, tricolor0.g, tricolor0.b,
        tricolor1.r, tricolor1.g, tricolor1.b,
        tricolor2.r, tricolor2.g, tricolor2.b
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

}


__global__
void linear_loss_kernel(int* tids,
                        int* pids,
                        int num_jobs,
                        Image* image,
                        TriMesh* mesh,
                        LinearFragment* colors,
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
    Color tricolor0 = colors[tri_id].c0;
    Color tricolor1 = colors[tri_id].c1;
    Color tricolor2 = colors[tri_id].c2;

    int h = image->cols;
    // Run generated teg function.
    auto outval = teg_linear_loss(
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

    // Accumulate derivatives.
    // TODO: There needs to be an easier way to accumulate derivatives..
    // Accumulate image.
    atomicAdd(&loss_image[pixel_id], outval);

}

__global__
void linear_deriv_kernel(int* tids,
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

    auto tri_id = tids[idx];
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

}



__global__
void linear_deriv_kernel_nodelta(int* tids,
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

    auto tri_id = tids[idx];
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
    auto outvals = teg_linear_deriv_nodelta(
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

}