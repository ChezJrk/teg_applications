#ifndef TYPES_CUH
#define TYPES_CUH

struct Color {
    float r;
    float g;
    float b;
};

struct DColor {
    float r;
    float g;
    float b;

    __device__ __host__ void operator=(const float in) {
        r = in;
        g = in;
        b = in;
    }
};

struct Vertex {
    float x;
    float y;
};

struct Vec2D {
    float x;
    float y;
};

struct DVec2D {
    float d_x;
    float d_y;

    __device__ __host__ void operator=(const float in) {
        d_x = in;
        d_y = in;
    }
};

struct DVertex {
    float x;
    float y;

    __device__ __host__ void operator=(const float in) {
        x = in;
        y = in;
    }
};

struct Image {
    int rows;
    int cols;
    Color* colors;
};

struct Grid {
    int rows;
    int cols;
    Vec2D* vecs;
    Color* colors;
};

struct DGrid {
    int rows;
    int cols;
    DVec2D* d_vecs;
    DColor* d_colors;
};

#endif