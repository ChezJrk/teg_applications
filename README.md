# Teg Applications
This is a research artifact for the paper: **Systematically Optimizing Parametric Discontinuities**. This repository contains the implementation for the applications of our differentiable programming language Teg. The applications include image stylization, fitting shader parameters, trajectory optimization, and optimizing physical designs. The implementation for Teg can be found at [https://github.com/ChezJrk/Teg](https://github.com/ChezJrk/Teg).

## Setup
To install the necessary Python packages run:
```
pip install -r requirements.txt
```

## Optimizing Physical Design
We study optimization of discontinuous parameters that control a physical design.
We consider an idealized series of bungee cords that deform.
After deformation, a string prevents further extension of the spring.
We jointly minimize the time and acceleration of a person connected to this bungee-string system.
We optimize the spring constants $k_1, k_2$ and the lengths of the strings $l_1, l_2$. 
We add the hard constraint that the person does not hit the ground to prevent death.

To optimize using derivatives computed by Teg run:
```
python3 physics/springs.py
```

To optimize using Teg, but ignoring the derivative contribution from the discontinuities (as would be done in PyTorch, TensorFlow, etc.) run:
```
python3 physics/springs.py --ignore_deltas
```

To optimize using derivatives estimated using finite differences run:
```
python3 physics/springs.py --finite_diff
```

Running each of these should take more than a few minutes.

## Image triangulation
This application approximates images with a colored triangle mesh, by formulating and minimizing an energy function over possible triangulations.
Our implementation generates a device kernel for the energy of a single triangle-pixel pair and its derivative using the Teg differentiable programming language. We then call this function from a CUDA kernel for maximum performance.

Compile the device code using (requires a CUDA toolkit installation): 
```
mkdir build && cd build
cmake ..
make
```
(The build process also applies the necessary code transformations and compiles Teg functions, and therefore can take some time to complete.)

To run the triangulation applications, use the following commands:

(Constant color fragments)
```
./triangulate_const <image_file> <tri-grid nx> <tri-grid ny> <use_deltas: y/n>
```

(Linear color fragments)
```
./triangulate_linear <image_file> <tri-grid nx> <tri-grid ny> <use_deltas: y/n>
```

(Quadratic color fragments)
```
./triangulate_linear <image_file> <tri-grid nx> <tri-grid ny> <use_deltas: y/n>
```

## Thresholded Perlin shader optimization
This application optimized the parameters of a thresholded Perlin shader to match a certain reference or guide image.
We model such a thresholded shader by first generating Perlin noise (read more [https://adrianb.io/2014/08/09/perlinnoise.html](here))
and then colorizing the region with negative and positive noise values with a specific color (C_+ and C_- in the paper).

For simplicity, we first calculate the value of the noise function at the four corners of each pixel and use bilinear interpolation to calculate noise values at continuous points within each pixel.

As is the case for triangulation, the shader program is compiled for a single pixel and paralleized through a CUDA kernel.

Compile the device code using (requires a CUDA toolkit installation): 
```
mkdir graphics/perlin_noise/build && cd graphics/perlin_noise/build
cmake ..
make
```

To run the noise optimization (from the build folder) for the two-tone shader (two colors, one for positive and one for negative space)
```
./optimize_perlin <image> <grid-nx> <grid-ny> <seed> <learning-rate>
```
grid-nx and grid-ny must perfectly divide the width and height respectively. Use seed=0 and learning-rate=1e-3 for default values.

For another version of the shader that added a per-pixel color map for added representative power, use
```
./optimize_perlin_colorized <image> <grid-nx> <grid-ny> <seed> <learning-rate>
```
