#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

struct Color {
    float r;
    float g;
    float b;
};

struct DColor {
    float r;
    float g;
    float b;
};

struct Vertex {
    float x;
    float y;
};

struct DVertex {
    float x;
    float y;

    __device__ __host__ void operator=(const float in) {
        x = in;
        y = in;
    }
};

struct ConstFragment{
    Color c;
    __device__ __host__ void operator=(const float in) {
        c = Color{in, in, in};
    }
    __device__ __host__ void operator=(const Color in) {
        c = in;
    }
};

struct DConstFragment{
    DColor d_c;
    __device__ __host__ void operator=(const float in) {
        d_c = DColor{in, in, in};
    }
    __device__ __host__ void operator=(const DColor d_in) {
        d_c = d_in;
    }
};

struct LinearFragment{
    Color c0;
    Color c1;
    Color c2;

    __device__ __host__ void operator=(const float in) {
        c0 = Color{in, in, in};
        c1 = Color{in, in, in};
        c2 = Color{in, in, in};
    }
    __device__ __host__ void operator=(const Color in) {
        c0 = in;
        c1 = in;
        c2 = in;
    }
};

struct DLinearFragment{
    DColor d_c0;
    DColor d_c1;
    DColor d_c2;

    __device__ __host__ void operator=(const float in) {
        d_c0 = DColor{in, in, in};
        d_c1 = DColor{in, in, in};
        d_c2 = DColor{in, in, in};
    }
    __device__ __host__ void operator=(const DColor in) {
        d_c0 = in;
        d_c1 = in;
        d_c2 = in;
    }
};

struct QuadraticFragment{
    Color c0; // vertex-colors
    Color c1;
    Color c2;
    Color c0h; // half-colors
    Color c1h;
    Color c2h;

    __device__ __host__ void operator=(const float in) {
        c0 = Color{in, in, in};
        c1 = Color{in, in, in};
        c2 = Color{in, in, in};
        c0h = Color{in, in, in};
        c1h = Color{in, in, in};
        c2h = Color{in, in, in};
    }
    __device__ __host__ void operator=(const Color in) {
        c0 = in;
        c1 = in;
        c2 = in;
        c0h = in;
        c1h = in;
        c2h = in;
    }
};

struct DQuadraticFragment{
    DColor d_c0; // vertex-colors
    DColor d_c1;
    DColor d_c2;
    DColor d_c0h; // half-colors
    DColor d_c1h;
    DColor d_c2h;

    __device__ __host__ void operator=(const float in) {
        d_c0 = DColor{in, in, in};
        d_c1 = DColor{in, in, in};
        d_c2 = DColor{in, in, in};
        d_c0h = DColor{in, in, in};
        d_c1h = DColor{in, in, in};
        d_c2h = DColor{in, in, in};
    }
    __device__ __host__ void operator=(const DColor in) {
        d_c0 = in;
        d_c1 = in;
        d_c2 = in;
        d_c0h = in;
        d_c1h = in;
        d_c2h = in;
    }
};

struct AABB {
    Vertex max;
    Vertex min;
};

struct Triangle {
    int a;
    int b;
    int c;

    __device__ __host__ AABB aabb(Vertex* v) {
        Vertex a = v[this->a];
        Vertex b = v[this->b];
        Vertex c = v[this->c];
        return AABB{Vertex{(a.x > b.x) ? ((a.x > c.x) ? a.x : c.x) : ((b.x > c.x) ? b.x : c.x),
                           (a.y > b.y) ? ((a.y > c.y) ? a.y : c.y) : ((b.y > c.y) ? b.y : c.y)},
                    Vertex{(a.x < b.x) ? ((a.x < c.x) ? a.x : c.x) : ((b.x < c.x) ? b.x : c.x),
                           (a.y < b.y) ? ((a.y < c.y) ? a.y : c.y) : ((b.y < c.y) ? b.y : c.y)}};
    }
};

struct SplineTriangle {
    int a;
    int b;
    int c;

    int ah;
    int bh;
    int ch;

    __device__ __host__ AABB aabb(Vertex* v) {
        // TODO: Figure this out.
    }
};

struct TriMesh {
    int num_triangles;
    Triangle* triangles;
    int num_vertices;
    Vertex* vertices;
    float* weights;

    __device__ __host__ Vertex* tv0(int i) {
        return vertices + triangles[i].a;
    }

    __device__ __host__ Vertex* tv1(int i) {
        return vertices + triangles[i].b;
    }

    __device__ __host__ Vertex* tv2(int i) {
        return vertices + triangles[i].c;
    }

    __device__ __host__ float wt(int i) {
        return weights[i];
    }
};

struct DTriMesh {
    TriMesh* trimesh;
    int num_vertices;
    DVertex* d_vertices;

    __device__ __host__ DVertex* tv0(int i) {
        return d_vertices + trimesh->triangles[i].a;
    }

    __device__ __host__ DVertex* tv1(int i) {
        return d_vertices + trimesh->triangles[i].b;
    }

    __device__ __host__ DVertex* tv2(int i) {
        return d_vertices + trimesh->triangles[i].c;
    }

};

struct Image {
    int rows;
    int cols;
    Color* colors;
};

#endif