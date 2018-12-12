cmake_minimum_required(VERSION 3.10)
project(mpi_lab_1 C)

set(CMAKE_C_STANDARD 90)

set(CMAKE_C_COMPILER mpicc)

add_executable(mpi-lab-3
        main.c)