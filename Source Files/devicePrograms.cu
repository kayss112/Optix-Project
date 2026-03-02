#include <optix.h>
#include <optix_device.h>
#include "../Header Files/LuanchParams.h"
extern "C" __constant__ LaunchParams optixLaunchParams;

extern "C" __global__ void __miss__radiance(){}
extern "C" __global__ void __anyhit__radiance() {}
extern "C" __global__ void __closesthit__radiance(){}

//实际渲染在这发生
extern "C" __global__ void __raygen__renderFrame() {
	const int frameID = optixLaunchParams.frameID;
	if (frameID == 0 &&
		optixGetLaunchIndex().x == 0 &&
		optixGetLaunchIndex().y == 0) {
		printf("############################################\n");
		printf("Hello world from OptiX 7 raygen program!\n(within a %ix%i-sized launch)\n",
			optixLaunchParams.fbSize.x,
			optixLaunchParams.fbSize.y);
		printf("############################################\n");
	}
	//----------------------------------------
	//渲染简单重复示例
	//----------------------------------

	// compute a test pattern based on pixel ID
	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;

	const int r = ((ix + frameID) % 256);
	const int g = ((iy + frameID) % 256);
	const int b = ((ix + iy + frameID) % 256);

	// convert to 32-bit rgba value (we explicitly set alpha to 0xff
	// to make stb_image_write happy ...
	const uint32_t rgba = 0xff000000
		| (r << 0) | (g << 8) | (b << 16);

	// and write to frame buffer ...
	const uint32_t fbIndex = ix + iy * optixLaunchParams.fbSize.x;
	optixLaunchParams.colorBuffer[fbIndex] = rgba;
}