include_directories(${CMAKE_SOURCE_DIR}/kinfu/include ${CMAKE_SOURCE_DIR}/kinfu/include/kinfu)

# Not recommended by CMake to use GLOB
# file(GLOB srcs *.cpp *.hpp)

# APP 01: OpenNI capture
#if (OPENNI_FOUND)
#    add_executable(demo demo.cpp)
#    target_link_libraries(demo ${OpenCV_LIBS} kfusion)

#    set_target_properties(demo PROPERTIES
#        DEBUG_POSTFIX "d"
#        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
#        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

#    install(TARGETS demo RUNTIME DESTINATION bin COMPONENT main)
#    install(FILES demo.cpp DESTINATION app COMPONENT main)
#endif(OPENNI_FOUND)



find_package(freenect2  REQUIRED)
include_directories(${freenect2_INCLUDE_DIRS})
message(" frenect2 ${freenect2_LIBRARY}")
# APP 02: BinGrabber capture
add_executable(kinfuMultiCam kinfuMultiCam.cpp)
target_link_libraries(kinfuMultiCam ${OpenCV_LIBS}  ${freenect2_LIBRARY} ${PCL_LIBRARIES} kinfu  )

add_executable(kinfuMultiCamGPUTexturing kinfuMultiCamGPUTexturing.cpp )
target_link_libraries(kinfuMultiCamGPUTexturing  ${OpenCV_LIBS}  ${freenect2_LIBRARY} ${PCL_LIBRARIES} kinfu  )



set_target_properties(kinfuMultiCam PROPERTIES
        DEBUG_POSTFIX "d"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

install(TARGETS kinfuMultiCam RUNTIME DESTINATION bin COMPONENT main)

set_target_properties(kinfuMultiCamGPUTexturing PROPERTIES
        DEBUG_POSTFIX "d"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

install(TARGETS kinfuMultiCamGPUTexturing RUNTIME DESTINATION bin COMPONENT main)

#install(FILES demo_bin.cpp DESTINATION app COMPONENT main)


#add_executable(multiCamRecorder multiCamRecorder.cpp)
#target_link_libraries(multiCamRecorder ${OpenCV_LIBS}  ${freenect2_LIBRARY} ${PCL_LIBRARIES}  kinfu )


#set_target_properties(multiCamRecorder PROPERTIES
#        DEBUG_POSTFIX "d"
#        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
#        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

#install(TARGETS multiCamRecorder RUNTIME DESTINATION bin COMPONENT main)
