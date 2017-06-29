SET(PROJECT_NAME MSC_THESIS)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

#set(BUILD_TYPE Release)
set(BUILD_TYPE RelWithDebugInfo)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# uncomment the following line and comment line afterward if you want to pick up the environmental variable instead of hardcoding the path
#set(DSO_PATH $ENV{DSO_PATH})
set(DSO_PATH "/home/daniel/git/dso")

enable_testing()

message("\n\n---- LOOKING FOR DSO at\n\"${DSO_PATH}\"")

find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread program_options filesystem)
find_package(LibZip)
find_package(OpenCV)
find_library(DSO_LIBRARY dso ${DSO_PATH}/build/lib)
find_package(PCL REQUIRED COMPONENTS common io features)
find_package(GeometricBurstiness CONFIG REQUIRED) # this exports GeometricBurstinessHeaders

# fix unnecessary link required by libproj in ubuntu 16.04
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
list(REMOVE_ITEM PCL_COMMON_LIBRARIES "vtkproj4")
list(REMOVE_ITEM PCL_IO_LIBRARIES "vtkproj4")
list(REMOVE_ITEM PCL_FEATURES_LIBRARIES "vtkproj4")

# march (and SSE?) flags are needed to prevent eigen from segfaulting. TODO: why?
set(CMAKE_CXX_FLAGS
   #"${SSE_FLAGS} -std=c++14 -O2"
   "${SSE_FLAGS} -g -O3 -std=c++14 -march=native -Wall"
)

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${DSO_PATH}/src
    ${DSO_PATH}/thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
)

# decide if we have ziplib.
if (LIBZIP_LIBRARY)
	message("--- found ziplib (${LIBZIP_VERSION}), compiling with zip capability.")
	add_definitions(-DHAS_ZIPLIB=1)
else()
	message("--- not found ziplib (${LIBZIP_LIBRARY}), compiling without zip capability.")
	set(LIBZIP_LIBRARY "")
endif()

add_executable(dso_pointcloud_generator
    src/MapGenerator.h

    src/main_generate_pointclouds.cpp
    src/MapGenerator.cpp
)

target_link_libraries(dso_pointcloud_generator
    ${DSO_LIBRARY}
    ${OpenCV_LIBS}
    ${LIBZIP_LIBRARY}
    boost_system
    boost_thread
    ${PCL_COMMON_LIBRARIES}
    ${PCL_IO_LIBRARIES}
)

add_executable(pfh_from_ply_test
    src/main_pfh_from_ply.cpp
)

target_link_libraries(pfh_from_ply_test
    ${PCL_COMMON_LIBRARIES}
    ${PCL_IO_LIBRARIES}
    ${PCL_FEATURES_LIBRARIES}
)


add_executable(main_run_semidense
    src/MapGenerator.h

    src/main_run_semidense.cpp
    src/MapGenerator.cpp
)

target_link_libraries(main_run_semidense
    ${DSO_LIBRARY}
    ${OpenCV_LIBS}
    ${LIBZIP_LIBRARY}
    boost_system
    boost_thread
    boost_program_options
    ${PCL_COMMON_LIBRARIES}
    ${PCL_IO_LIBRARIES}
)


add_executable(main_analyze_relocalization
    src/Relocalization.h
    src/LargeBagOfWords.h
    src/MapGenerator.h
    src/FusedFeatureDescriptors.h
    src/Datasets.h
    
    src/LargeBagOfWords.cpp
    src/Relocalization.cpp
    src/MapGenerator.cpp
    src/FusedFeatureDescriptors.cpp
    src/Datasets.cpp
    src/main_analyze_relocalization.cpp
)

target_link_libraries(main_analyze_relocalization
    ${DSO_LIBRARY}
    ${LIBZIP_LIBRARY}
    ${OpenCV_LIBS}
    boost_filesystem
    boost_program_options
    ${PCL_COMMON_LIBRARIES}
    ${PCL_IO_LIBRARIES}
    GeometricBurstinessHeaders
)


add_executable(create_descriptor_training_data
    src/LargeBagOfWords.h
    src/Relocalization.h
    src/MapGenerator.h
    src/Datasets.h
    
    src/LargeBagOfWords.cpp
    src/Relocalization.cpp
    src/MapGenerator.cpp
    src/Datasets.cpp
    src/create_descriptor_training_data.cpp
)

target_link_libraries(create_descriptor_training_data
    ${DSO_LIBRARY}
    ${LIBZIP_LIBRARY}
    ${OpenCV_LIBS}
    boost_filesystem
    boost_program_options
    ${PCL_COMMON_LIBRARIES}
    ${PCL_IO_LIBRARIES}
    GeometricBurstinessHeaders
)

add_executable(test_approx_kmeans
    src/LargeBagOfWords.h
    
    src/LargeBagOfWords.cpp
    test/test_approx_kmeans.cpp
)

target_link_libraries(test_approx_kmeans
    ${OpenCV_LIBS}
)
add_test(TestApproxKmeans test_approx_kmeans)


