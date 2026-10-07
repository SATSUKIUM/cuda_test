cmake_minimum_required(VERSION 3.18)

project(cuda_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_ARCHITECTURES 86)

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

# ソースファイル
file(GLOB SRC_FILES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.cu"
)

# Executable
foreach(src_file IN LISTS SRC_FILES)

    get_filename_component(exe_name "${src_file}" NAME_WE)

    # 実行ファイル
    add_executable(${exe_name} "${src_file}")

    # インクルードヘッダ
    target_include_directories(${exe_name} PRIVATE "${CMAKE_SOURCE_DIR}/include")

    # Link ROOT libraries
    target_link_libraries(${exe_name}
        PRIVATE
        ROOT::Hist
        ROOT::RIO
        ROOT::Tree
        ROOT::Graf
        ROOT::Gpad
    )

    # Set output directory for executables
    set_target_properties(${exe_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Install
    install(TARGETS ${exe_name}
        RUNTIME DESTINATION .
    )

endforeach()