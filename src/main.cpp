#include "sparse_slotmap.hpp"
#include "dense_slotmap.hpp"
#include <iostream>

int main()
{
	sparse_slotmap::slot_map<std::string> strings;
	auto key = strings.emplace("Hello");

	dense_slotmap::slot_map<std::string> denseStrings;
	auto denseKey = denseStrings.emplace("World");

	std::cout << "Hello, World!" << std::endl;
    return 0;
}