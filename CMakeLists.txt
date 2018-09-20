cmake_minimum_required(VERSION 3.6)

set(CMAKE_CXX_STANDARD 14)

project(landmarks)

# OpenCV
find_package(OpenCV 4.0.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# libxml2
# sudo apt-get install libxml2-dev
#find_package(LibXml2 REQUIRED)
#include_directories(${LIBXML2_INCLUDE_DIR})

add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(landmarks ${OpenCV_LIBS})
#target_link_libraries(train ${LIBXML2_LIBRARIES})
