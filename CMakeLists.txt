cmake_minimum_required(VERSION 3.1)

project(HandDigit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_BUILD_TYPE "Release")


set(EXECUTABLE_OUTPUT_PATH ..)

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

find_package(Boost
    COMPONENTS
      program_options
      unit_test_framework
      serialization
    REQUIRED
)


#Find armadillo package
find_package(Armadillo REQUIRED)
set(ML_LIB armadillo mlpack)

#Image processing and camera shits
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(Source)

set(Source
    source/main.cpp    
)

# Build Executable
add_executable(${PROJECT_NAME} ${Source})

# Link modules
target_link_libraries(${PROJECT_NAME} ${ML_LIB} ${OpenCV_LIBS} stdc++fs)

# Install the executable
install(TARGETS ${PROJECT_NAME} DESTINATION ${EXECUTABLE_OUTPUT_PATH})



# Find includes in corresponding build directories
# set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Instruct CMake to run moc automatically when needed.
# set(CMAKE_AUTOMOC ON)
