#include "triangle.cuh"

int generate_jobs(int image_width, int image_height, 
                  TriMesh* trimesh,
                  int* tids, int* pids ) {
    int job_count = 0;
    for (int tid = 0; tid < trimesh->num_triangles; tid++) {

        AABB aabb = trimesh->triangles[tid].aabb(trimesh->vertices);

        int max_x = std::ceil(aabb.max.x) + 2; //(int)(std::ceil( std::max(vx0, std::max(vx1, vx2)))) + 1;
        int max_y = std::ceil(aabb.max.y) + 2; //(int)(std::ceil( std::max(vy0, std::max(vy1, vy2)))) + 1;
        int min_x = std::floor(aabb.min.x) - 2; //(int)(std::floor( std::min(vx0, std::min(vx1, vx2)))) - 1;
        int min_y = std::floor(aabb.min.y) - 2; //(int)(std::floor( std::min(vy0, std::min(vy1, vy2)))) - 1;

        if (min_x < 0) min_x = 0;
        if (min_y < 0) min_y = 0;
        if (max_x >= image_width) max_x = image_width - 1;
        if (max_y >= image_height) max_y = image_height - 1;

        for (int tx = min_x; tx <= max_x; tx++) {
            for (int ty = min_y; ty <= max_y; ty++) {
                tids[job_count] = tid;
                pids[job_count] = tx * image_height + ty;
                //if(job_count % 100000 == 0) std::cout << job_count << " " << tid << " " << max_x << " " << max_y << " " << min_x << " " << min_y << std::endl;
                job_count ++;
            }
        }
    }

    return job_count;
}

float sgn(float x){
    return (x > 0) ? 1: -1;
}

int compute_triangle_regularization(TriMesh* mesh,
                                    DTriMesh* d_mesh) {

    for (int tid = 0; tid < mesh->num_triangles; tid++) {
        Vertex* a = mesh->tv0(tid);
        Vertex* b = mesh->tv1(tid);
        Vertex* c = mesh->tv2(tid);

        // Cross-product for area
        float area = ((a->x - b->x)*(a->y - c->y) - (a->x - c->x)*(a->y - b->y)) / 1.f;

        DVertex* d_a = d_mesh->tv0(tid);
        DVertex* d_b = d_mesh->tv1(tid);
        DVertex* d_c = d_mesh->tv2(tid);

        d_a->x += (-1.0/area) * (b->y - c->y);
        d_a->y += (-1.0/area) * (c->x - b->x);

        d_b->x += (-1.0/area) * (c->y - a->y);
        d_b->y += (-1.0/area) * (a->x - c->x);

        d_c->x += (-1.0/area) * (a->y - b->y);
        d_c->y += (-1.0/area) * (b->x - a->x);

    }

}

int build_initial_triangles(TriMesh* trimesh,
                             int nx, int ny,
                             int image_width, int image_height) {

    float tri_width  = (float)(image_width)  / nx;
    float tri_height = (float)(image_height) / ny;

    cudaMallocManaged(&trimesh->vertices, sizeof(Vertex) * (nx + 1) * (ny + 1));
    cudaMallocManaged(&trimesh->weights, sizeof(float) * (nx + 1) * (ny + 1));
    int num_vertices = 0;
    for(int i = 0; i < nx + 1; i++) {
        for(int j = 0; j < ny + 1; j++) {
            Vertex* a = trimesh->vertices + (i * (ny + 1) + j);
            a->x = tri_width  * i;
            a->y = tri_height * j;
            trimesh->weights[(i * (ny + 1) + j)] = !((i == 0) || (j == 0) || (i == nx) || (j == ny));
            num_vertices ++;
        }
    }

    cudaMallocManaged(&trimesh->triangles, sizeof(Triangle) * (nx) * (ny) * 2);

    int num_triangles = 0;
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
            trimesh->triangles[2 * (i * ny + j)] = Triangle{((i + 0) * (ny + 1) + j + 0),
                                                       ((i + 0) * (ny + 1) + j + 1),
                                                       ((i + 1) * (ny + 1) + j + 1)};

            trimesh->triangles[2 * (i * ny + j) + 1] = Triangle{((i + 0) * (ny + 1) + j + 0),
                                                           ((i + 1) * (ny + 1) + j + 1),
                                                           ((i + 1) * (ny + 1) + j + 0)};

            num_triangles += 2;
        }
    }

    trimesh->num_triangles = num_triangles;
    trimesh->num_vertices = num_vertices;

    std::cout << "Built triangles " << num_triangles << ", " << num_vertices << std::endl;
    return num_triangles;
}


template<typename T>
__global__ 
void set_zero(T* data, int max_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < max_id)
        data[idx] = 0;
}


template<typename T>
__global__ 
void set_value(T* data, T val, int max_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < max_id)
        data[idx] = val;
}


__host__
void build_d_mesh(TriMesh* trimesh,
                  DTriMesh** d_trimesh) {

    cudaMallocManaged(d_trimesh, sizeof(DTriMesh));
    cudaMallocManaged(&((*d_trimesh)->d_vertices), sizeof(Vertex) * trimesh->num_vertices);
    (*d_trimesh)->trimesh = trimesh;
    (*d_trimesh)->num_vertices = trimesh->num_vertices;
    set_zero<DVertex><<<((trimesh->num_vertices) / 256) + 1, 256>>>((*d_trimesh)->d_vertices, trimesh->num_vertices);
    cudaDeviceSynchronize();
}

__global__
void update_vertices(TriMesh* mesh,
                     DTriMesh* d_mesh,
                     float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > mesh->num_vertices)
        return;

    Vertex* a = mesh->vertices + idx;
    DVertex* d_a = d_mesh->d_vertices + idx;
    a->x = a->x - alpha * d_a->x * mesh->weights[idx];
    a->y = a->y - alpha * d_a->y * mesh->weights[idx];

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