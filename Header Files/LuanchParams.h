#pragma once
#include "gdt/gdt.h"
#include "gdt/math/vec.h"
struct LaunchParams {
	gdt::vec2i fbSize;
	uint32_t* colorBuffer;
	int frameID{ 0 };
};