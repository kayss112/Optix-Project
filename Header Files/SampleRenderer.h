#pragma once
#include "../Header Files/LuanchParams.h"
#include "../Header Files/optix7.h"
#include "../Header Files/CUDABuffer.h"
#include "gdt/math/AffineSpace.h"
#include <vector>
struct Camera {
	gdt::vec3f from;
	gdt::vec3f at;
	gdt::vec3f up;
};
struct TriangleMesh
{
	void addUnitCube(const gdt::affine3f& xfm);

	void addCube(const gdt::vec3f& center, const gdt::vec3f& size);

	std::vector<gdt::vec3f> vertex;
	std::vector<gdt::vec3i> index;
};
class SampleRenderer {
public:
	SampleRenderer(const TriangleMesh& model);
	//画一帧
	void render();
	//定义帧缓存
	void resize(const gdt::vec2i& newSize);
	//现在渲染的颜色缓存
	void downloadPixels(uint32_t h_pixels[]);
	/*! set camera to render with */
	void setCamera(const Camera& camera);
protected:
	//初始化Optix和检查1错误
	void InitOptix();
	//创建上下文？
	void createContext();
	/*! creates the module that contains all the programs we are going
  to use. in this simple example, we use a single module from a
  single .cu file, using a single embedded ptx string */
	void createModule();
	//为raygen创建所有设置
	void createRaygenPrograms();
	//为未命中创建
	void createMissPrograms();
	//创建命中逻辑
	void createHitgroupPrograms();
	//集合所有的渲染管线
	void createPipeline();
	//构建SBT
	void buildSBT();

	/*! build an acceleration structure for the given triangle mesh */
	OptixTraversableHandle buildAccel(const TriangleMesh& model);
protected:
	CUcontext cudaContext;
	CUstream stream;
	cudaDeviceProp deviceProps;

	//给设备的上下文
	OptixDeviceContext optixContext;
	//管线
	OptixPipeline pipeline;
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	OptixPipelineLinkOptions pipelineLinkOptions = {};
	//包含的模型
	OptixModule module;
	OptixModuleCompileOptions moduleCompileOptions = {};

	//用于程序的vector和SBT
	std::vector<OptixProgramGroup> raygenPGs;
	CUDABuffer raygenRecordsBuffer;
	std::vector<OptixProgramGroup> missPGs;
	CUDABuffer missRecordsBuffer;
	std::vector<OptixProgramGroup> hitgroupPGs;
	CUDABuffer hitgroupRecordsBuffer;
	OptixShaderBindingTable sbt = {};
	//host的参数设置,还有缓存给设备用于存储
	LaunchParams launchParams;
	CUDABuffer launchParamsBuffer;

	CUDABuffer colorBuffer;
	//我们要渲染的相机
	Camera lastSetCamera;

	//我们要渲染的模型
	const TriangleMesh model;
	CUDABuffer vertexBuffer;
	CUDABuffer indexBuffer;
	//存放(final,compacted)加速结构
	CUDABuffer asBuffer;
};
