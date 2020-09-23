cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(transformer-app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# set(CMAKE_BUILD_TYPE Debug)

add_executable(transformer-app transformer_models.cpp gpt_local_pipeline.cpp gpt_test.cpp)
target_link_libraries(transformer-app "${TORCH_LIBRARIES}")
set_property(TARGET transformer-app PROPERTY CXX_STANDARD 14)
