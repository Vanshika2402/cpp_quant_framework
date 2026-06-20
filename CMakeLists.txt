cmake_minimum_required(VERSION 3.16)
project(JapanDerivativesCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(japan_derivatives src/japan_derivatives.cpp)

target_compile_options(japan_derivatives PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W4>
)
