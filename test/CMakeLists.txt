function(create_test test_name)
    file(RELATIVE_PATH REL_BINARY_DIR
         ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE floatx gtest_main)
    add_test(NAME ${REL_BINARY_DIR}/${test_name} COMMAND ${TEST_TARGET_NAME})
endfunction(create_test)


add_library(ieee_helper STATIC IEEEHelper.cpp)

create_test(conversion)
create_test(assignment)
create_test(rel_ops)
create_test(arithmetic)
create_test(stream)
create_test(std_integration)
create_test(NanInf)
create_test(round_nearest)
create_test(value_representation)
target_link_libraries(test_value_representation PRIVATE ieee_helper)
create_test(value_representation_half)
target_link_libraries(test_value_representation_half PRIVATE ieee_helper)
create_test(value_representation_bits)
target_link_libraries(test_value_representation_bits PRIVATE ieee_helper)
