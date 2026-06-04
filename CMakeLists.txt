cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME slotmaps)

project(${PROJECT_NAME}
  VERSION 0.1.0
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
  add_compile_options(/permissive- /Zc:preprocessor /Zc:__cplusplus)
endif()

file(GLOB_RECURSE SRC CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
)

add_executable(${PROJECT_NAME} ${SRC})

target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if (MSVC)
  source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src FILES ${SRC})
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION .)