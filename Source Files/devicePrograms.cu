#include <optix.h>
#include <optix_device.h>
#include "../Header Files/LuanchParams.h"
extern "C" __constant__ LaunchParams optixLaunchParams;
//有单个光线类型
enum {SURFACE_RAY_TYPE=0,RAY_TYPE_COUNT};

static __forceinline__ __device__
void* unpackPointer(uint32_t i0, uint32_t i1) {
	const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
	void* ptr = reinterpret_cast<void*>(uptr);
	return ptr;
}

static __forceinline__ __device__
void packPointer(void* ptr, uint32_t& i0, uint32_t& i1) {
	const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T* getPRD() {
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

extern "C" __global__ void __miss__radiance(){
	gdt::vec3f& prd = *(gdt::vec3f*)getPRD<gdt::vec3f>();
	prd = gdt::vec3f(1.f);
}
extern "C" __global__ void __anyhit__radiance() {}
extern "C" __global__ void __closesthit__radiance(){
	const int primID = optixGetPrimitiveIndex();
	gdt::vec3f& prd = *(gdt::vec3f*)getPRD<gdt::vec3f>();
	prd = gdt::randomColor(primID);
}

//实际渲染在这发生
extern "C" __global__ void __raygen__renderFrame() {
	//计算测试pattern
	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;

	const auto& camera = optixLaunchParams.camera;
	// our per-ray data for this example. what we initialize it to
	// won't matter, since this value will be overwritten by either
	// the miss or hit program, anyway
	gdt::vec3f pixelColorPRD = gdt::vec3f(0.f);

	//储存我们的PRD
	uint32_t u0, u1;
	packPointer(&pixelColorPRD, u0, u1);

	//标准化
	const gdt::vec2f screen(gdt::vec2f(ix + .5f,iy + .5f) / gdt::vec2f(optixLaunchParams.frame.size));

	//生成光方向
	gdt::vec3f rayDir = normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);

	optixTrace(optixLaunchParams.traversable,
		camera.position,
		rayDir,
		0.f,    // tmin
		1e20f,  // tmax
		0.0f,   // rayTime
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
		SURFACE_RAY_TYPE,             // SBT offset
		RAY_TYPE_COUNT,               // SBT stride
		SURFACE_RAY_TYPE,             // missSBTIndex 
		u0, u1);

	const int r = int(255.99f * pixelColorPRD.x);
	const int g = int(255.99f * pixelColorPRD.y);
	const int b = int(255.99f * pixelColorPRD.z);

	// convert to 32-bit rgba value (we explicitly set alpha to 0xff
	// to make stb_image_write happy ...
	const uint32_t rgba = 0xff000000
		| (r << 0) | (g << 8) | (b << 16);

	// and write to frame buffer ...
	const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
	optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}