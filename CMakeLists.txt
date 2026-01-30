cmake_minimum_required(VERSION 3.16)
project(LHN_AI LANGUAGES CXX)

add_library(LHN_AI INTERFACE)

target_include_directories(LHN_AI
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(lensing_experiment experiments/lensing.cpp)
target_link_libraries(lensing_experiment PRIVATE LHN_AI)
