add_executable(example example.cpp)
target_link_libraries(example PRIVATE floatx)

add_executable(example2 example2.cpp)
target_link_libraries(example2 PRIVATE floatx)

add_executable(common_type common_type.cpp)
target_link_libraries(common_type PRIVATE floatx)

add_executable(DemoNewton DemoNewton.cpp)
target_link_libraries(DemoNewton PRIVATE floatx)