cmake_minimum_required(VERSION 3.16)
project(PixelClusterDetector CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Build type ────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ── Compiler flags ────────────────────────────────────────────────────────────
# Every architecture flag is PROBED before use (check_cxx_compiler_flag) rather
# than hard-coded, so an unsupported one degrades to "no tuning" instead of
# failing the build. This is what lets the same file serve Apple Silicon, MSVC
# on x86-64, and Linux/GCC without editing.
include(CheckCXXCompilerFlag)

option(DETECTOR_NATIVE_ARCH
       "Tune the binary for the CPU that builds it (disable for portable binaries)"
       ON)

function(add_arch_flag_if_supported)
  # Adds the FIRST flag the compiler accepts, then stops.
  foreach(flag ${ARGN})
    string(MAKE_C_IDENTIFIER "HAVE${flag}" var)
    check_cxx_compiler_flag("${flag}" ${var})
    if(${var})
      add_compile_options("${flag}")
      message(STATUS "Architecture tuning: ${flag}")
      return()
    endif()
  endforeach()
  message(STATUS "Architecture tuning: none available — generic build")
endfunction()

if(MSVC)
  add_compile_options(/O2 /W3)
  if(DETECTOR_NATIVE_ARCH)
    # AVX2 auto-vectorises the Welford inner loop. It is a *compile*-time check
    # only: the resulting binary needs an AVX2-capable CPU (Intel Haswell 2013 /
    # AMD Excavator 2015 and later). Configure with -DDETECTOR_NATIVE_ARCH=OFF
    # to target an older machine.
    add_arch_flag_if_supported(/arch:AVX2)
  endif()
else()
  add_compile_options(-O3 -Wall -Wextra)
  if(DETECTOR_NATIVE_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|arm")
      # ARM (Apple Silicon, Raspberry Pi, ARM servers): -mcpu carries both the
      # ISA and the tuning; -march=native is rejected by Apple clang.
      add_arch_flag_if_supported(-mcpu=native -mcpu=apple-m1)
    else()
      # x86-64 and everything else: -march=native is the portable spelling.
      add_arch_flag_if_supported(-march=native -mtune=native)
    endif()
  endif()
endif()

# ── OpenCV ────────────────────────────────────────────────────────────────────
# Set OpenCV_DIR if cmake can't find it automatically, e.g.:
#   cmake -DOpenCV_DIR=C:/opencv/build/x64/vc16/lib ..
#
# It is that subfolder, not build/ itself, that holds OpenCVConfig.cmake on a
# prebuilt Windows OpenCV. Use vc17 instead of vc16 for the newer toolsets.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# ── Sources ───────────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/BackgroundModel.cpp
    src/ClusterDetector.cpp
  src/DataExporter.cpp
)

add_executable(detector ${SOURCES})
# The headers live next to the sources in src/ — there is no include/ directory
# (the old value pointed at a path that has never existed; it worked only
# because #include "X.h" resolves relative to the including file).
target_include_directories(detector PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(detector PRIVATE ${OpenCV_LIBS})

# ── OpenMP ────────────────────────────────────────────────────────────────────
# Optional by construction: BackgroundModel.cpp guards <omp.h> with #ifdef
# _OPENMP, and a compiler that does not know "#pragma omp" simply ignores it.
# Losing OpenMP therefore costs speed in the Welford loop, never correctness.
#
# The previous condition was `OpenMP_CXX_FOUND AND NOT APPLE`, whose else-branch
# hard-coded Homebrew paths — so ANY machine where find_package failed (a Linux
# box, or Windows without the MSVC OpenMP component) got Apple-only flags fed to
# a non-Apple compiler. Order it properly instead: standard target first, the
# Homebrew shim only on Apple, a warning everywhere else.
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: ${OpenMP_CXX_FLAGS} (standard CMake target)")
    target_link_libraries(detector PRIVATE OpenMP::OpenMP_CXX)
elseif(APPLE)
    # Apple clang ships no OpenMP runtime; Homebrew's libomp supplies it. The
    # prefix differs between Apple Silicon (/opt/homebrew) and Intel
    # (/usr/local), so ask brew and fall back to both well-known locations.
    execute_process(COMMAND brew --prefix libomp
                    OUTPUT_VARIABLE BREW_LIBOMP
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    find_path(LIBOMP_INCLUDE_DIR omp.h
              HINTS "${BREW_LIBOMP}/include"
                    /opt/homebrew/opt/libomp/include
                    /usr/local/opt/libomp/include)
    find_library(LIBOMP_LIBRARY omp
                 HINTS "${BREW_LIBOMP}/lib"
                       /opt/homebrew/opt/libomp/lib
                       /usr/local/opt/libomp/lib)
    if(LIBOMP_INCLUDE_DIR AND LIBOMP_LIBRARY)
        message(STATUS "OpenMP: Homebrew libomp -> ${LIBOMP_LIBRARY}")
        target_compile_options(detector PRIVATE -Xpreprocessor -fopenmp)
        target_include_directories(detector PRIVATE "${LIBOMP_INCLUDE_DIR}")
        target_link_libraries(detector PRIVATE "${LIBOMP_LIBRARY}")
    else()
        message(WARNING "OpenMP not found (try: brew install libomp) — "
                        "building single-threaded. Correct, just slower.")
    endif()
else()
    message(WARNING "OpenMP not found — building single-threaded. "
                    "Correct, just slower.")
endif()

# ── pthreads (needed for std::thread on Linux/macOS) ─────────────────────────
find_package(Threads REQUIRED)
target_link_libraries(detector PRIVATE Threads::Threads)

# ── Windows: put the runtime DLLs next to the .exe ───────────────────────────
# Windows searches the executable's own directory before anything else, so
# copying OpenCV's DLLs there makes detector.exe start however it is launched:
# from the GUI, from a terminal, by double-click, or by a future script. Without
# this it depends on PATH being prepared first, and when that is missing the
# failure is silent and baffling -- the process dies before main() with exit
# code 0xC0000135 (STATUS_DLL_NOT_FOUND), printing nothing at all.
if(WIN32)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21)
    add_custom_command(TARGET detector POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              $<TARGET_RUNTIME_DLLS:detector> $<TARGET_FILE_DIR:detector>
      COMMAND_EXPAND_LISTS
      COMMENT "Copying runtime DLLs next to detector.exe")
  else()
    message(WARNING "CMake < 3.21: cannot auto-copy the OpenCV DLLs next to "
                    "detector.exe. Add <OpenCV_DIR>/x64/vc16/bin to PATH, or "
                    "launch the detector through PipelineController.py, which "
                    "sets that up itself.")
  endif()
endif()

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS detector RUNTIME DESTINATION bin)
