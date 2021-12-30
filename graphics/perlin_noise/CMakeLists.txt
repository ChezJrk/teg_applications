cmake_minimum_required(VERSION 3.10)

# set the project name
project(perlin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}) 

set(CUDA_NVCC_FLAGS_DEBUG "-g")
set(CUDA_NVCC_FLAGS_RELEASE "-O3")

# Credits: https://stackoverflow.com/users/1593077/einpoklum
# Auto-detect GPU compute capability
if (NOT CUDA_TARGET_COMPUTE_CAPABILITY)
    if("$ENV{CUDA_SM}" STREQUAL "")
        set(ENV{CUDA_INCLUDE_DIRS} "${CUDA_INCLUDE_DIRS}")
        set(ENV{CUDA_CUDART_LIBRARY} "${CUDA_CUDART_LIBRARY}")
        set(ENV{CMAKE_CXX_COMPILER} "${CMAKE_CXX_COMPILER}")
        execute_process(COMMAND 
            bash -c "${CMAKE_CURRENT_SOURCE_DIR}/scripts/get_cuda_sm.sh" 
            OUTPUT_VARIABLE CUDA_TARGET_COMPUTE_CAPABILITY_)
    else()
        set(CUDA_TARGET_COMPUTE_CAPABILITY_ $ENV{CUDA_SM})
    endif()

    set(CUDA_TARGET_COMPUTE_CAPABILITY "${CUDA_TARGET_COMPUTE_CAPABILITY_}" 
        CACHE STRING "CUDA compute capability of the (first) CUDA device on \
        the system, in XY format (like the X.Y format but no dot); see table \
        of features and capabilities by capability X.Y value at \
        https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications")

    execute_process(COMMAND 
        bash -c "echo -n $(echo ${CUDA_TARGET_COMPUTE_CAPABILITY})" 
        OUTPUT_VARIABLE CUDA_TARGET_COMPUTE_CAPABILITY)
    execute_process(COMMAND 
        bash -c "echo ${CUDA_TARGET_COMPUTE_CAPABILITY} | sed 's/^\\([0-9]\\)\\([0-9]\\)/\\1.\\2/;' | xargs echo -n" 
        OUTPUT_VARIABLE FORMATTED_COMPUTE_CAPABILITY)

    message(STATUS 
        "CUDA device-side code will assume compute capability \
        ${FORMATTED_COMPUTE_CAPABILITY}")
endif()

set(CUDA_GENCODE
    "arch=compute_${CUDA_TARGET_COMPUTE_CAPABILITY}, code=compute_${CUDA_TARGET_COMPUTE_CAPABILITY}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode ${CUDA_GENCODE} )
# ------------------------------

set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

execute_process(COMMAND python3 -m teg --includes
    OUTPUT_VARIABLE TEG_INCLUDE_DIRS)
include_directories("${TEG_INCLUDE_DIRS}")

add_executable(render_perlin render_perlin.cu)
add_executable(optimize_perlin optimize_perlin.cu)
add_executable(optimize_perlin_2t optimize_perlin_2t.cu)
add_executable(optimize_perlin_colorized optimize_perlin_colorized.cu)

add_custom_target(teg
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_thresholded.h"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_noise.h"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_threshold_deriv.h"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_threshold_fwdderiv.h"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_double_threshold_fwdderiv.h"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_double_thresholded.h"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_colorized_thresholded.h"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_colorized_threshold_fwdderiv.h")

add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_thresholded.h"
    COMMAND python3 -m teg --compile "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_thresholded.py" -f single -t CUDA_C -o "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_thresholded.h" -m teg_perlin_thresholded
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_thresholded.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Teg kernels: teg_perlin_thresholded.py"
)

add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_noise.h"
    COMMAND python3 -m teg --compile "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_noise.py" -f single -t CUDA_C -o "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_noise.h" -m teg_perlin_noise
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_noise.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Teg kernels: teg_perlin_noise.py"
)

add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_threshold_deriv.h"
    COMMAND python3 -m teg --compile "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_threshold_deriv.py" -f single -t CUDA_C -o "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_threshold_deriv.h" -m teg_perlin_threshold_deriv
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_threshold_deriv.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Teg kernels: teg_perlin_threshold_deriv.py"
)

add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_threshold_fwdderiv.h"
    COMMAND python3 -m teg --compile "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_threshold_fwdderiv.py" -f single -t CUDA_C -o "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_threshold_fwdderiv.h" -m teg_perlin_threshold_fwdderiv
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_threshold_fwdderiv.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Teg kernels: teg_perlin_threshold_fwdderiv.py"
)

add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_double_thresholded.h"
    COMMAND python3 -m teg --compile "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_double_thresholded.py" -f single -t CUDA_C -o "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_double_thresholded.h" -m teg_perlin_double_thresholded
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_double_thresholded.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Teg kernels: teg_perlin_double_thresholded.py"
)

add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_double_threshold_fwdderiv.h"
    COMMAND python3 -m teg --compile "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_double_threshold_fwdderiv.py" -f single -t CUDA_C -o "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_double_threshold_fwdderiv.h" -m teg_perlin_double_threshold_fwdderiv
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_double_threshold_fwdderiv.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Teg kernels: teg_perlin_double_threshold_fwdderiv.py"
)

add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_colorized_thresholded.h"
    COMMAND python3 -m teg --compile "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_colorized_thresholded.py" -f single -t CUDA_C -o "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_colorized_thresholded.h" -m teg_perlin_colorized_thresholded
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_colorized_thresholded.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Teg kernels: teg_perlin_colorized_thresholded.py"
)

add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_colorized_threshold_fwdderiv.h"
    COMMAND python3 -m teg --compile "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_colorized_threshold_fwdderiv.py" -f single -t CUDA_C -o "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_colorized_threshold_fwdderiv.h" -m teg_perlin_colorized_threshold_fwdderiv
    MAIN_DEPENDENCY "${CMAKE_CURRENT_SOURCE_DIR}/teg_perlin_colorized_threshold_fwdderiv.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Teg kernels: teg_perlin_colorized_threshold_fwdderiv.py"
)


set_target_properties(render_perlin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_dependencies(render_perlin teg)
target_link_libraries(render_perlin ${OpenCV_LIBRARIES})


set_target_properties(optimize_perlin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_dependencies(optimize_perlin teg)
target_link_libraries(optimize_perlin ${OpenCV_LIBRARIES})


set_target_properties(optimize_perlin_2t PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_dependencies(optimize_perlin_2t teg)
target_link_libraries(optimize_perlin_2t ${OpenCV_LIBRARIES})

set_target_properties(optimize_perlin_colorized PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_dependencies(optimize_perlin_colorized teg)
target_link_libraries(optimize_perlin_colorized ${OpenCV_LIBRARIES})