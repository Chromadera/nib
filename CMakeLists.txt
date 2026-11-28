cmake_minimum_required(VERSION 3.21)
project(nib VERSION 0.4.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation path")
set(HIP_COMPILER "${ROCM_PATH}/bin/hipcc" CACHE FILEPATH "HIP compiler")
set(GPU_TARGET "gfx1100" CACHE STRING "AMD RDNA3/RDNA4 GPU target(s), semicolon-separated (gfx1100-gfx1103, gfx1200, gfx1201)")

set(_NIB_SUPPORTED_TARGETS gfx1100 gfx1101 gfx1102 gfx1103 gfx1200 gfx1201)
set(_NIB_OFFLOAD_FLAGS "")
set(_NIB_GPU_TARGETS_CLEAN "")
foreach(_tgt IN LISTS GPU_TARGET)
    string(STRIP "${_tgt}" _tgt)
    if(_tgt STREQUAL "")
        continue()
    endif()
    list(FIND _NIB_SUPPORTED_TARGETS "${_tgt}" _idx)
    if(_idx EQUAL -1)
        message(FATAL_ERROR
            "Unsupported GPU_TARGET '${_tgt}'. Supported AMD RDNA3/RDNA4 targets: "
            "gfx1100, gfx1101, gfx1102, gfx1103, gfx1200, gfx1201.")
    endif()
    list(APPEND _NIB_OFFLOAD_FLAGS "--offload-arch=${_tgt}")
    list(APPEND _NIB_GPU_TARGETS_CLEAN "${_tgt}")
endforeach()
if(NOT _NIB_OFFLOAD_FLAGS)
    message(FATAL_ERROR "GPU_TARGET must specify at least one supported RDNA3/RDNA4 target.")
endif()
string(JOIN "," NIB_GPU_TARGET_STR ${_NIB_GPU_TARGETS_CLEAN})
message(STATUS "Nib GPU target(s): ${NIB_GPU_TARGET_STR}")

# RDNA4 FP8 WMMA prefill kernels (src/kernels_wmma_fp8_rdna4.hip) are built only when a gfx12
# target is requested; otherwise the host-only src/kernels_wmma_fp8_stub.cpp provides the same API
# and reports the kernels as not built (host code checks rdna4_fp8_prefill_supported() at runtime).
# Deliberately no compile definition: the existing HIP objects keep byte-identical commands.
set(NIB_HAVE_RDNA4_FP8 0)
foreach(_tgt IN LISTS _NIB_GPU_TARGETS_CLEAN)
    if(_tgt STREQUAL "gfx1200" OR _tgt STREQUAL "gfx1201")
        set(NIB_HAVE_RDNA4_FP8 1)
    endif()
endforeach()
message(STATUS "Nib RDNA4 FP8 WMMA prefill kernels: ${NIB_HAVE_RDNA4_FP8}")

set(NIB_GIT_SHA "unknown")
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE _nib_git_sha
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_nib_git_sha)
        set(NIB_GIT_SHA "${_nib_git_sha}")
    endif()
endif()

function(add_hip_object target_name hip_src)
    set(obj_file "${CMAKE_CURRENT_BINARY_DIR}/${target_name}.o")
    add_custom_command(
        OUTPUT "${obj_file}"
        COMMAND "${HIP_COMPILER}"
            -O3
            -fPIC
            ${_NIB_OFFLOAD_FLAGS}
            -std=c++17
            -Wall -Wextra -Wno-unused-parameter -Wno-unused-function
            -D__HIP_PLATFORM_AMD__
            -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I"${CMAKE_CURRENT_SOURCE_DIR}/src"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/${hip_src}"
            -o "${obj_file}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/${hip_src}"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/nib_arch.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/inkling_engine.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/sparse_global.h"
            ${ARGN}
        COMMENT "Compiling RDNA HIP kernel ${hip_src} (${NIB_GPU_TARGET_STR})"
    )
    set(${target_name}_OBJ "${obj_file}" PARENT_SCOPE)
endfunction()

add_hip_object(repack_soa            "src/repack_soa.hip")
add_hip_object(kernels_core          "src/kernels_core.hip")
add_hip_object(kernels_fa            "src/kernels_fa_relbias.hip")
add_hip_object(kernels_gemv_soa      "src/kernels_gemv_soa.hip")
add_hip_object(kernels_wmma          "src/kernels_wmma_prefill.hip")
add_hip_object(kernels_vision        "src/kernels_vision.hip")
add_hip_object(kernels_sparse_global "src/kernels_sparse_global.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/sparse_global.h")
add_hip_object(kernels_decode_fused  "src/kernels_decode_fused.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/kernels_decode_fused.h")
add_hip_object(kernels_fa_decode_v2  "src/kernels_fa_decode_v2.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/kernels_fa_decode_v2.h")
add_hip_object(kernels_fa_decode_v3  "src/kernels_fa_decode_v3.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/kernels_fa_decode_v3.h")
add_hip_object(kernels_fa_decode_v4  "src/kernels_fa_decode_v4.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/kernels_fa_decode_v4.h")
add_hip_object(kernels_moe_v3        "src/kernels_moe_v3.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/kernels_moe_v3.h")
add_hip_object(kernels_step7h_fused  "src/kernels_step7h_fused.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/kernels_step7h_fused.h")
add_hip_object(kernels_step7i_fused  "src/kernels_step7i_fused.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/kernels_step7i_fused.h")
add_hip_object(kernels_step7j_fdot2  "src/kernels_step7j_fdot2.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/kernels_step7j_fdot2.h")
add_hip_object(kernels_mtp_verify    "src/kernels_mtp_verify.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/kernels_mtp_verify.h")
add_hip_object(kernels_dspark_mtp    "src/kernels_dspark_mtp.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/dspark_mtp.h")
add_hip_object(kernels_verify_batch  "src/kernels_verify_batch.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/kernels_verify_batch.h")

add_hip_object(repack_iq             "src/repack_iq.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/iq_types.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/iq_tables.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/iq_dequant_device.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/kernels_iq.h")
add_hip_object(kernels_moe_iq        "src/kernels_moe_iq.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/iq_types.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/iq_tables.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/iq_dequant_device.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/kernels_iq.h")
add_hip_object(kernels_dflash2_mtp   "src/kernels_dflash2_mtp.hip"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/dflash2_mtp_kernels.h")

if(NIB_HAVE_RDNA4_FP8)
    add_hip_object(kernels_wmma_fp8  "src/kernels_wmma_fp8_rdna4.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/nib_arch_fp8.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/kernels_wmma_fp8.h")
    add_hip_object(kernels_wmma_fp8_iq "src/kernels_wmma_fp8_iq_rdna4.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/nib_arch_fp8.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/kernels_wmma_fp8.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/iq_types.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/iq_tables.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/iq_dequant_device.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/kernels_iq.h")
endif()

set(NIB_HIP_OBJECTS
    ${repack_soa_OBJ}
    ${kernels_core_OBJ}
    ${kernels_fa_OBJ}
    ${kernels_gemv_soa_OBJ}
    ${kernels_wmma_OBJ}
    ${kernels_vision_OBJ}
    ${kernels_sparse_global_OBJ}
    ${kernels_decode_fused_OBJ}
    ${kernels_fa_decode_v2_OBJ}
    ${kernels_fa_decode_v3_OBJ}
    ${kernels_fa_decode_v4_OBJ}
    ${kernels_moe_v3_OBJ}
    ${kernels_step7h_fused_OBJ}
    ${kernels_step7i_fused_OBJ}
    ${kernels_step7j_fdot2_OBJ}
    ${kernels_mtp_verify_OBJ}
    ${kernels_dspark_mtp_OBJ}
    ${kernels_verify_batch_OBJ}
    ${repack_iq_OBJ}
    ${kernels_moe_iq_OBJ}
    ${kernels_dflash2_mtp_OBJ}
)
set(NIB_FP8_HOST_SOURCES "")
if(NIB_HAVE_RDNA4_FP8)
    list(APPEND NIB_HIP_OBJECTS ${kernels_wmma_fp8_OBJ} ${kernels_wmma_fp8_iq_OBJ})
else()
    set(NIB_FP8_HOST_SOURCES src/kernels_wmma_fp8_stub.cpp)
endif()

add_custom_target(nib_hip_objs DEPENDS ${NIB_HIP_OBJECTS})

add_executable(nib
    src/gguf_loader.cpp
    src/engine.cpp
    src/dspark_mtp.cpp
    src/dflash2_mtp.cpp
    src/vision_encoder.cpp
    src/server.cpp
    src/main.cpp
    ${NIB_FP8_HOST_SOURCES}
    ${NIB_HIP_OBJECTS}
)
add_dependencies(nib nib_hip_objs)

target_include_directories(nib PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${ROCM_PATH}/include"
)

target_compile_definitions(nib PRIVATE
    __HIP_PLATFORM_AMD__
    NIB_VERSION="${PROJECT_VERSION}"
    NIB_GIT_SHA="${NIB_GIT_SHA}"
    NIB_GPU_TARGET="${NIB_GPU_TARGET_STR}"
)
target_compile_options(nib PRIVATE -O3 -march=native -Wall -Wextra -Wno-unused-parameter)

target_link_directories(nib PRIVATE "${ROCM_PATH}/lib")
target_link_libraries(nib PRIVATE amdhip64 pthread dl)

set_target_properties(nib PROPERTIES
    BUILD_RPATH "${ROCM_PATH}/lib"
    INSTALL_RPATH "${ROCM_PATH}/lib"
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rdna_kernels.hip")
    # tests/test_rdna4_fp8.hip (RDNA4 FP8 suites 10-11) is included by test_rdna_kernels.hip.
    add_hip_object(test_rdna_kernels "tests/test_rdna_kernels.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rdna4_fp8.hip"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/nib_arch_fp8.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/kernels_wmma_fp8.h")
    add_executable(nib_kernel_test
        ${test_rdna_kernels_OBJ}
        ${NIB_FP8_HOST_SOURCES}
        ${NIB_HIP_OBJECTS}
    )
    add_dependencies(nib_kernel_test nib_hip_objs)
    set_target_properties(nib_kernel_test PROPERTIES LINKER_LANGUAGE CXX)
    # Only used by NIB_FP8_HOST_SOURCES (the host stub); the HIP objects carry their own flags.
    target_include_directories(nib_kernel_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src" "${ROCM_PATH}/include")
    target_compile_definitions(nib_kernel_test PRIVATE __HIP_PLATFORM_AMD__)
    target_link_directories(nib_kernel_test PRIVATE "${ROCM_PATH}/lib")
    target_link_libraries(nib_kernel_test PRIVATE amdhip64 pthread dl)
    set_target_properties(nib_kernel_test PROPERTIES
        BUILD_RPATH "${ROCM_PATH}/lib"
        INSTALL_RPATH "${ROCM_PATH}/lib"
    )
endif()
