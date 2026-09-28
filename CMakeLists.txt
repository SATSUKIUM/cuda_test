cmake_minimum_required(VERSION 3.18)

project(cuda_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# デフォルトのインストール先

# -DCMAKE_INSTALL_PREFIX=... で変更可能

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/bin"
CACHE PATH "Install path" FORCE)
endif()

# ROOT

find_package(ROOT REQUIRED COMPONENTS Hist RIO Tree Graf Gpad)

# ROOTのライブラリディレクトリをRPATHに設定

set(CMAKE_BUILD_RPATH "${ROOT_LIBRARY_DIR}")
set(CMAKE_INSTALL_RPATH "${ROOT_LIBRARY_DIR}")

# Executable

add_executable(test src/test1.cu)

# ROOT libraries

target_link_libraries(test PRIVATE ROOT::Hist ROOT::RIO ROOT::Tree ROOT::Graf ROOT::Gpad)

# Build output

set_target_properties(test PROPERTIES
RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Install

install(TARGETS test
RUNTIME DESTINATION .
)
