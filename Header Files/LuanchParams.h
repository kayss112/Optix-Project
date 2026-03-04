#pragma once
#include "gdt/gdt.h"
#include "gdt/math/vec.h"
#include "../Header Files/optix7.h"
struct LaunchParams {
	struct 
	{
		gdt::vec2i size;
		uint32_t* colorBuffer;
	}frame;
	struct 
	{
		gdt::vec3f position;
		gdt::vec3f direction;
		gdt::vec3f horizontal;
		gdt::vec3f vertical;
	}camera;

	OptixTraversableHandle traversable;
};