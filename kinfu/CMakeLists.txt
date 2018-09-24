function(default_properties target)
  set_target_properties(${target} PROPERTIES
    DEBUG_POSTFIX "d"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    if (NOT ${target} MATCHES "^test_")
      install(TARGETS ${the_target} RUNTIME DESTINATION ".")
    endif()
endfunction()


set(module_name kinfu)

message("${module_name}")



set(Source_srcs
 	src/error.cpp
 	src/initialization.cpp
    src/device_memory.cpp
    src/tsdf_volume.cpp
    src/marching_cubes.cpp
    src/cyclical_buffer.cpp
    src/kinfu.cpp
   )

set(Source_cuda_srcs
    src/cuda/device.hpp
    src/cuda/utils.hpp
    src/cuda/pointer_shift.cu
    src/cuda/utils.hpp
    src/cuda/tsdf_volume.cu
    src/cuda/push.cu
    src/cuda/marching_cubes.cu
    src/cuda/bilateral_pyrdown.cu
    src/cuda/maps.cu
    src/cuda/normals_eigen.cu
    src/cuda/texture_mapping.cu
 )

include_directories(include include/kinfu/ src src/cuda )
set(__has_cuda ON)
#check_cuda(__has_cuda)

set(__lib_type STATIC)
if (${ARGV1} MATCHES "SHARED|STATIC")
    set(__lib_type ${ARGV1})
endif()

cuda_add_library(${module_name} ${__lib_type} ${Source_srcs} ${Source_cuda_srcs} )
message ("STATUS" ${CMAKE_BINARY_DIR})

if(MSVC)
    set_target_properties(${module_name} PROPERTIES DEFINE_SYMBOL KFUSION_API_EXPORTS)
else()
    add_definitions(-DKFUSION_API_EXPORTS)
endif()

default_properties(${module_name})

#set_target_properties(${module_name} PROPERTIES INSTALL_NAME_DIR lib)

install(TARGETS ${module_name}
        RUNTIME DESTINATION bin COMPONENT main
        LIBRARY DESTINATION lib COMPONENT main
        ARCHIVE DESTINATION lib COMPONENT main)

install(DIRECTORY include/ DESTINATION include/ FILES_MATCHING PATTERN "*.h*")

target_link_libraries(kinfu ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY}  ${OpenCV_LIBS} )
