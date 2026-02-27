cmake_minimum_required(VERSION 3.5)
project(Subsense)

# Forzamos C++11 y optimización
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")

# ESTA ES LA MAGIA: Busca OpenCV donde sea que esté instalado en el sistema
find_package(OpenCV REQUIRED)

set(SS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if(APPLE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_INSTALL_NAME_DIR ${CMAKE_INSTALL_PREFIX}/lib)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

file(GLOB LIB_SRC_FILES ${SS_DIR}/Subsense/*.cpp)

# Le decimos al compilador que incluya las rutas dinámicas de OpenCV
include_directories(${SS_DIR} ${OpenCV_INCLUDE_DIRS})

add_library(Subsense SHARED ${LIB_SRC_FILES})

# Linkeamos con las librerías dinámicas detectadas por CMake
target_link_libraries(Subsense ${OpenCV_LIBS})