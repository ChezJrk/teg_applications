cmake_minimum_required(VERSION 3.10)

# set the project name
project(triangulate LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}) 

set(CMAKE_CUDA_FLAGS "--gpu-architecture=compute_75 --gpu-code=sm_75")
set(CUDA_NVCC_FLAGS_DEBUG "-g")
set(CUDA_NVCC_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

execute_process(COMMAND python3 -m teg --includes
    OUTPUT_VARIABLE TEG_INCLUDE_DIRS)
include_directories("${TEG_INCLUDE_DIRS}")

add_executable(triangulate_const triangulate_const.cu)
add_executable(triangulate_linear triangulate_linear.cu)
add_executable(triangulate_quadratic triangulate_quadratic.cu)
add_executable(test test.cu)

add_custom_target(teg
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/renderpixel.h"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tegpixel.h"

    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/teg_linear_deriv.h"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/teg_linear_integral.h"

    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/teg_linear_bilinear_deriv.h"
    
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/teg_quadratic_deriv.h"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/teg_quadratic_integral.h")

add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/renderpixel.h"
    COMMAND python3 -m teg --compile "${CMAKE_CURRENT_SOURCE_DIR}/integral.py" -f single -t CUDA_C -o "${CMAKE_CURRENT_SOURCE_DIR}/renderpixel.h" -m renderpixel
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/integral.py"
    COMMENT "Building Teg kernels: integral.py"
)

add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/tegpixel.h"
    COMMAND python3 -m teg --compile "${CMAKE_CURRENT_SOURCE_DIR}/loss_function.py" -f single -t CUDA_C -o "${CMAKE_CURRENT_SOURCE_DIR}/tegpixel.h" -m tegpixel
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/loss_function.py"
    COMMENT "Building Teg kernels: loss_function.py"
)

add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/teg_linear_deriv.h"
    COMMAND python3 -m teg --compile "${CMAKE_CURRENT_SOURCE_DIR}/teg_linear_deriv.py" -f single -t CUDA_C -o "${CMAKE_CURRENT_SOURCE_DIR}/teg_linear_deriv.h" -m teg_linear_deriv
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/teg_linear_deriv.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Teg kernels: teg_linear_deriv.py"
)

add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/teg_linear_integral.h"
    COMMAND python3 -m teg --compile "${CMAKE_CURRENT_SOURCE_DIR}/teg_linear_integral.py" -f single -t CUDA_C -o "${CMAKE_CURRENT_SOURCE_DIR}/teg_linear_integral.h" -m teg_linear_integral
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/teg_linear_integral.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Teg kernels: teg_linear_integral.py"
)

add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/teg_quadratic_deriv.h"
    COMMAND python3 -m teg --compile "${CMAKE_CURRENT_SOURCE_DIR}/teg_quadratic_deriv.py" -f single -t CUDA_C -o "${CMAKE_CURRENT_SOURCE_DIR}/teg_quadratic_deriv.h" -m teg_quadratic_deriv
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/teg_quadratic_deriv.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Teg kernels: teg_quadratic_deriv.py"
)

add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/teg_quadratic_integral.h"
    COMMAND python3 -m teg --compile "${CMAKE_CURRENT_SOURCE_DIR}/teg_quadratic_integral.py" -f single -t CUDA_C -o "${CMAKE_CURRENT_SOURCE_DIR}/teg_quadratic_integral.h" -m teg_quadratic_integral
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/teg_quadratic_integral.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Teg kernels: teg_quadratic_integral.py"
)


add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/teg_linear_bilinear_deriv.h"
    COMMAND python3 -m teg --compile "${CMAKE_CURRENT_SOURCE_DIR}/teg_linear_bilinear_deriv.py" -f single -t CUDA_C -o "${CMAKE_CURRENT_SOURCE_DIR}/teg_linear_bilinear_deriv.h" -m teg_linear_bilinear_deriv
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/teg_linear_bilinear_deriv.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Teg kernels: teg_linear_bilinear_deriv.py"
)

set_target_properties(triangulate_const PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_dependencies(triangulate_const teg)
target_link_libraries(triangulate_const ${OpenCV_LIBRARIES})

set_target_properties(triangulate_linear PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_dependencies(triangulate_linear teg)
target_link_libraries(triangulate_linear ${OpenCV_LIBRARIES})

set_target_properties(triangulate_quadratic PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_dependencies(triangulate_quadratic teg)
target_link_libraries(triangulate_quadratic ${OpenCV_LIBRARIES})

set_target_properties(test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_dependencies(test teg)
target_link_libraries(test ${OpenCV_LIBRARIES})