message(STATUS "Including the PI API ROCM backend.")

 # cannot rely on cmake support for ROCM; it assumes runtime API is being used.
 # we only require the ROCM driver API to be used
 # rocm_rocm_LIBRARY variable defines the path to libhsa-runtime64.so, the ROCM Driver API library.

#find_package(ROCM 4.0 REQUIRED)

# Make imported library global to use it within the project.

set(ROCM_INCLUDE_DIRS "/opt/rocm/hip/include")
set(hsa_inc_dir "/opt/rocm/hsa/include")

add_library(pi_rocm SHARED
   "${sycl_inc_dir}/CL/sycl/detail/pi.h"
   "${sycl_inc_dir}/CL/sycl/detail/pi.hpp"
   "pi_rocm.hpp"
   "pi_rocm.cpp" 
)


add_dependencies(sycl-toolchain pi_rocm)

set_target_properties(pi_rocm PROPERTIES LINKER_LANGUAGE CXX)


target_include_directories(pi_rocm
        PRIVATE
        ${sycl_inc_dir}
        ${sycl_plugin_dir}
        ${ROCM_INCLUDE_DIRS}
        ${hsa_inc_dir}
)

target_link_libraries(pi_rocm PUBLIC OpenCL-Headers)
 
# if(AMD_ROCM)
#   set(ROCM_ROCM_LIBRARY "/opt/rocm/hip/lib/libamdhip64.so")
#   add_library(rocmdrv SHARED IMPORTED GLOBAL)
# 
# 
#   set_target_properties(
#     rocmdrv PROPERTIES 
#       IMPORTED_LOCATION             ${ROCM_ROCM_LIBRARY}
#       INTERFACE_INCLUDE_DIRECTORIES ${ROCM_INCLUDE_DIRS}
#   )
# 
#   target_link_libraries(pi_rocm PUBLIC rocmdrv)
#   target_compile_definitions(pi_rocm PRIVATE __HIP_PLATFORM_HCC__)
# else(NVIDIA_ROCM)
  find_package(CUDA 11.3 REQUIRED)

  add_library(cudadrv SHARED IMPORTED GLOBAL)
  
  set_target_properties(
    cudadrv PROPERTIES
      IMPORTED_LOCATION             ${CUDA_CUDA_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
  )

  target_link_libraries(pi_rocm PUBLIC OpenCL-Headers cudadrv)
  target_compile_definitions(pi_rocm PRIVATE __HIP_PLATFORM_NVCC__)
# endif()

add_common_options(pi_rocm)

install(TARGETS pi_rocm
  LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT pi_rocm
  RUNTIME DESTINATION "bin" COMPONENT pi_rocm
)
