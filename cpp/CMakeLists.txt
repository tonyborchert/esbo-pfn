# Generators for the five simulated control problems: furuta_gen, cartpole_gen, plants_gen.
#
#   cmake -S cpp -B cpp/build -DBLAS=accelerate|openblas|mkl [-DNATIVE=ON]
#   cmake --build cpp/build -j
#   cmake --install cpp/build --prefix "$PFN_HOME"        # -> $PFN_HOME/bin
#
# `pfn data build` runs exactly these steps. Only furuta_gen needs LAPACK (the scipy-identical
# discrete Riccati solver of the LQR design). Default BLAS: Accelerate on macOS, OpenBLAS
# elsewhere. MKL is linked sequentially (one std::thread per worker does the parallelism).
cmake_minimum_required(VERSION 3.16)
project(esbo_generators VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

option(NATIVE "Compile with -march=native" OFF)
set(BLAS "" CACHE STRING "LAPACK provider for furuta_gen: accelerate, openblas or mkl")
if(BLAS STREQUAL "")
  if(APPLE)
    set(BLAS accelerate)
  else()
    set(BLAS openblas)
  endif()
endif()
message(STATUS "esbo generators: BLAS=${BLAS} NATIVE=${NATIVE}")

find_package(Threads REQUIRED)

# No fused multiply-add contraction: the same source gives the same trajectories with GCC and clang.
set(ESBO_FLAGS -Wall -Wextra -ffp-contract=off)
if(NATIVE)
  list(APPEND ESBO_FLAGS -march=native)
endif()

function(esbo_generator name)
  add_executable(${name} src/${name}.cpp)
  target_include_directories(${name} PRIVATE include problems)
  target_compile_options(${name} PRIVATE ${ESBO_FLAGS})
  target_compile_definitions(${name} PRIVATE ESBO_GENERATOR_VERSION="${PROJECT_VERSION}")
  target_link_libraries(${name} PRIVATE Threads::Threads)
endfunction()

esbo_generator(furuta_gen)
esbo_generator(cartpole_gen)
esbo_generator(plants_gen)

if(BLAS STREQUAL "accelerate")
  target_link_libraries(furuta_gen PRIVATE "-framework Accelerate")
elseif(BLAS STREQUAL "mkl")
  if(NOT DEFINED ENV{MKLROOT})
    message(FATAL_ERROR "BLAS=mkl needs MKLROOT (module load imkl)")
  endif()
  set(MKLROOT $ENV{MKLROOT})
  foreach(cand "${MKLROOT}/lib/intel64" "${MKLROOT}/lib")
    if(EXISTS "${cand}/libmkl_core.so" OR EXISTS "${cand}/libmkl_core.a")
      set(MKL_LIBDIR "${cand}")
      break()
    endif()
  endforeach()
  if(NOT MKL_LIBDIR)
    message(FATAL_ERROR "no libmkl_core under ${MKLROOT}/lib or ${MKLROOT}/lib/intel64")
  endif()
  target_link_directories(furuta_gen PRIVATE "${MKL_LIBDIR}")
  target_link_options(furuta_gen PRIVATE "-Wl,-rpath,${MKL_LIBDIR}")
  target_link_libraries(furuta_gen PRIVATE -Wl,--start-group mkl_intel_lp64 mkl_sequential mkl_core
                        -Wl,--end-group m dl)
elseif(BLAS STREQUAL "openblas")
  find_library(OPENBLAS_LIB NAMES openblas openblas64_ PATHS ENV OPENBLAS_ROOT ENV EBROOTOPENBLAS
               PATH_SUFFIXES lib lib64)
  if(OPENBLAS_LIB)
    target_link_libraries(furuta_gen PRIVATE ${OPENBLAS_LIB})
  else()
    find_package(LAPACK REQUIRED)
    target_link_libraries(furuta_gen PRIVATE ${LAPACK_LIBRARIES})
  endif()
else()
  message(FATAL_ERROR "unknown BLAS '${BLAS}' (accelerate, openblas or mkl)")
endif()

install(TARGETS furuta_gen cartpole_gen plants_gen RUNTIME DESTINATION bin)
