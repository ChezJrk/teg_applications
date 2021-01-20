#include "triangle.cuh"

#include "teg_quadratic_integral.h"
#include "teg_quadratic_deriv.h"

__global__
void update_quadratic_colors(int num_colors,
                   QuadraticFragment* colors,
                   DQuadraticFragment* d_colors,
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

    Color* c0h = &colors[idx].c0h;
    DColor* d_c0h = &d_colors[idx].d_c0h;
    c0h->r = c0h->r - alpha * d_c0h->r;
    c0h->g = c0h->g - alpha * d_c0h->g;
    c0h->b = c0h->b - alpha * d_c0h->b;

    Color* c1h = &colors[idx].c1h;
    DColor* d_c1h = &d_colors[idx].d_c1h;
    c1h->r = c1h->r - alpha * d_c1h->r;
    c1h->g = c1h->g - alpha * d_c1h->g;
    c1h->b = c1h->b - alpha * d_c1h->b;

    Color* c2h = &colors[idx].c2h;
    DColor* d_c2h = &d_colors[idx].d_c2h;
    c2h->r = c2h->r - alpha * d_c2h->r;
    c2h->g = c2h->g - alpha * d_c2h->g;
    c2h->b = c2h->b - alpha * d_c2h->b;
}


__host__
void build_quadratic_colors(TriMesh* trimesh,
                         QuadraticFragment** colors) {

    cudaMallocManaged(colors, sizeof(QuadraticFragment) * trimesh->num_triangles);
    Color def = Color{0.5, 0.5, 0.5};
    set_value<QuadraticFragment><<<((trimesh->num_triangles) / 256) + 1, 256>>>(
                    *colors, QuadraticFragment{def, def, def, def, def, def}, trimesh->num_triangles);
    cudaDeviceSynchronize();

}

__host__
void build_d_quadratic_colors(TriMesh* trimesh,
                           DQuadraticFragment** d_colors) {

    cudaMallocManaged(d_colors, sizeof(DQuadraticFragment) * trimesh->num_triangles);
    set_zero<DQuadraticFragment><<<((trimesh->num_triangles) / 256) + 1, 256>>>(*d_colors, trimesh->num_triangles);
    cudaDeviceSynchronize();

}


__global__
void quadratic_integral_kernel(int* tids,
                      int* pids,
                      int num_jobs,
                      TriMesh* mesh,
                      QuadraticFragment* colors,
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

    Color tricolor0h = colors[tri_id].c0h;
    Color tricolor1h = colors[tri_id].c1h;
    Color tricolor2h = colors[tri_id].c2h;

    // Run generated teg function.
    auto outvals = teg_quadratic_integral(
        a->x,a->y,
        b->x,b->y,
        c->x,c->y,

        floorf(pixel_id / h),
        floorf(pixel_id / h) + 1,
        (float)(pixel_id % h),
        (float)(pixel_id % h + 1),

        tricolor0.r, tricolor0.g, tricolor0.b,
        tricolor1.r, tricolor1.g, tricolor1.b,
        tricolor2.r, tricolor2.g, tricolor2.b,

        tricolor0h.r, tricolor0h.g, tricolor0h.b,
        tricolor1h.r, tricolor1h.g, tricolor1h.b,
        tricolor2h.r, tricolor2h.g, tricolor2h.b
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
void quadratic_deriv_kernel(int* tids,
                             int* pids,
                             int num_jobs,
                             Image* image,
                             TriMesh* mesh,
                             DTriMesh* d_mesh,
                             QuadraticFragment* colors,
                             DQuadraticFragment* d_colors)
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

    Color tricolor0h = colors[tri_id].c0h;
    Color tricolor1h = colors[tri_id].c1h;
    Color tricolor2h = colors[tri_id].c2h;

    int h = image->cols;
    // Run generated teg function.
    auto outvals = teg_quadratic_deriv(
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

        tricolor0.r, tricolor0.g, tricolor0.b,
        tricolor1.r, tricolor1.g, tricolor1.b,
        tricolor2.r, tricolor2.g, tricolor2.b,

        tricolor0h.r, tricolor0h.g, tricolor0h.b,
        tricolor1h.r, tricolor1h.g, tricolor1h.b,
        tricolor2h.r, tricolor2h.g, tricolor2h.b
    );

    DVertex* d_a = d_mesh->tv0(tri_id);
    DVertex* d_b = d_mesh->tv1(tri_id);
    DVertex* d_c = d_mesh->tv2(tri_id);

    DQuadraticFragment *d_pcolor = d_colors + tri_id;

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

    atomicAdd(&d_pcolor->d_c0h.r, outvals[15]);
    atomicAdd(&d_pcolor->d_c0h.g, outvals[16]);
    atomicAdd(&d_pcolor->d_c0h.b, outvals[17]);

    atomicAdd(&d_pcolor->d_c1h.r, outvals[18]);
    atomicAdd(&d_pcolor->d_c1h.g, outvals[19]);
    atomicAdd(&d_pcolor->d_c1h.b, outvals[20]);

    atomicAdd(&d_pcolor->d_c2h.r, outvals[21]);
    atomicAdd(&d_pcolor->d_c2h.g, outvals[22]);
    atomicAdd(&d_pcolor->d_c2h.b, outvals[23]);

}