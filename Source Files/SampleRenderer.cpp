#include "../Header Files/SampleRenderer.h"
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>
//创建"_ptx"
extern "C" char devicePrograms_ptx[];

static void context_log_cb(unsigned int level,
	const char* tag,
	const char* message,
	void*)
{
	fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

//SBT记录
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT)char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	//假数据后续增加
	void* data;
};
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT)MissRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	//假数据
	void* data;
};
//打中时
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT)HitgroupRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT)char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	//假数据
	int objectID;
};
/*! constructor - performs all setup, including initializing
  optix, creates module, pipeline, programs, SBT, etc. */
SampleRenderer::SampleRenderer(const TriangleMesh& model)
{
	InitOptix();
	
	std::cout << "#osc: creating optix context ..." << std::endl;
	createContext();

	std::cout << "#osc: setting up module ..." << std::endl;
	createModule();

	std::cout << "#osc: creating raygen programs ..." << std::endl;
	createRaygenPrograms();
	std::cout << "#osc: creating miss programs ..." << std::endl;
	createMissPrograms();
	std::cout << "#osc: creating hitgroup programs ..." << std::endl;
	createHitgroupPrograms();

	launchParams.traversable = buildAccel(model);

	std::cout << "#osc: setting up optix pipeline ..." << std::endl;
	createPipeline();

	std::cout << "#osc: building SBT ..." << std::endl;
	buildSBT();

	launchParamsBuffer.alloc(sizeof(launchParams));
	std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;

	std::cout << GDT_TERMINAL_GREEN;
	std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
	std::cout << GDT_TERMINAL_DEFAULT;
}
//画一帧
void SampleRenderer::render()
{
	// sanity check: make sure we launch only after first resize is
// already done:
	if (launchParams.frame.size.x == 0) { return; }

	launchParamsBuffer.upload(&launchParams, 1);

	OPTIX_CHECK(optixLaunch(pipeline,stream,launchParamsBuffer.d_pointer(),
		launchParamsBuffer.sizeInBytes,&sbt,launchParams.frame.size.x,launchParams.frame.size.y,1));

	//SYNC--确保帧已经在我们下载前渲染好
	CUDA_SYNC_CHECK();
}
/*! resize frame buffer to given resolution */
void SampleRenderer::resize(const gdt::vec2i& newSize)
{
	if (newSize.x == 0 | newSize.y == 0) { return; }
	colorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));

	launchParams.frame.size = newSize;
	launchParams.frame.colorBuffer = (uint32_t*)colorBuffer.d_ptr;
}
//下载buffer
void SampleRenderer::downloadPixels(uint32_t h_pixels[])
{
	colorBuffer.download(h_pixels, launchParams.frame.size.x * launchParams.frame.size.y);
}
//设置相机参数
void SampleRenderer::setCamera(const Camera& camera)
{
	lastSetCamera = camera;
	launchParams.camera.position = camera.from;
	launchParams.camera.direction = gdt::normalize(camera.at - camera.from);
	const float cosFovy = 0.66f;
	const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);
	launchParams.camera.horizontal = cosFovy * aspect * gdt::normalize(gdt::cross(launchParams.camera.direction, camera.up));
	launchParams.camera.vertical = cosFovy * gdt::normalize(gdt::cross(launchParams.camera.horizontal, launchParams.camera.direction));
}

void SampleRenderer::InitOptix()
{
	cudaFree(0);
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if(numDevices==0){ throw std::runtime_error("AAAA no CUDA capable devices found!"); }
	std::cout << "found" << numDevices << "CUDA devices" << std::endl;

	OPTIX_CHECK(optixInit());
	std::cout << GDT_TERMINAL_GREEN
		<< "#osc: successfully initialized optix... yay!"
		<< GDT_TERMINAL_DEFAULT << std::endl;
}
//创建上下文，在本次中只给主GPU创建
void SampleRenderer::createContext()
{
	const int deviceID = 0;
	CUDA_CHECK(SetDevice(deviceID));
	CUDA_CHECK(StreamCreate(&stream));

	cudaGetDeviceProperties(&deviceProps, deviceID);
	std::cout << "runnig on device" << deviceProps.name << std::endl;

	CUresult cuRes = cuCtxGetCurrent(&cudaContext);
	if (cuRes != CUDA_SUCCESS) { fprintf(stderr, "Error querying current context: error code%d\n", cuRes); }

	OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
	OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
}
//创建module包含所有我们要用到的programs。这里使用单个module来自单个.cu文件
void SampleRenderer::createModule()
{
	moduleCompileOptions.maxRegisterCount = 50;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	pipelineCompileOptions = {};
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.numPayloadValues = 2;
	pipelineCompileOptions.numAttributeValues = 2;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

	pipelineLinkOptions.maxTraceDepth = 2;

	const std::string ptxCode = devicePrograms_ptx;

	char log[2048];
	size_t sizeof_log = sizeof(log);
#if OPTIX_VERSION>=70700
	OPTIX_CHECK(optixModuleCreate(optixContext,
		&moduleCompileOptions, &pipelineCompileOptions, ptxCode.c_str(), ptxCode.size(), log, &sizeof_log, &module));
#else
	OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
		&moduleCompileOptions,
		&pipelineCompileOptions,
		ptxCode.c_str(),
		ptxCode.size(),
		log,      // Log string
		&sizeof_log,// Log string sizse
		&module
	));
#endif
	if (sizeof_log > 1){PRINT(log);}
}
//给Raygen做一些设置
void SampleRenderer::createRaygenPrograms()
{
	//设置单个光线策略
	raygenPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc.raygen.module = module;
	pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";
	//OptixProgramGroup raypg;
	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &raygenPGs[0]));

	if (sizeof_log > 1)PRINT(log);
}
//设置未命中程序
void SampleRenderer::createMissPrograms()
{
	//只使用但根光线
	missPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	pgDesc.miss.module = module;
	pgDesc.miss.entryFunctionName = "__miss__radiance";

	//OptixProgramGroup raypg
	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log,
		&sizeof_log, &missPGs[0]));
	if (sizeof_log > 1) { PRINT(log); }
}
//设置命中程序
void SampleRenderer::createHitgroupPrograms()
{
	//只设置单根线
	hitgroupPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	//为什么用CH？
	pgDesc.hitgroup.moduleCH = module;
	pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	pgDesc.hitgroup.moduleAH = module;
	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &hitgroupPGs[0]));
	if (sizeof_log > 1) { PRINT(log); }
}
//集合所有的Programs
void SampleRenderer::createPipeline()
{
	std::vector<OptixProgramGroup> programGroups;
	for (auto pg : raygenPGs) { programGroups.push_back(pg); }
	for (auto pg : missPGs) { programGroups.push_back(pg); }
	for (auto pg : hitgroupPGs) { programGroups.push_back(pg); }

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixPipelineCreate(optixContext, &pipelineCompileOptions,
		&pipelineLinkOptions, programGroups.data(), (int)programGroups.size(), log, &sizeof_log,
		&pipeline));
	if (sizeof_log > 1) { PRINT(log); }

	OPTIX_CHECK(optixPipelineSetStackSize(pipeline, 2 * 1024, 2 * 1024, 2 * 1024, 1));
	if (sizeof_log > 1) { PRINT(log); }
}
//创建绑定表
void SampleRenderer::buildSBT()
{
	//------------------------------
	//创建raygen records
	//-----------------------------
	std::vector<RaygenRecord> raygenRecords;
	for (int i = 0; i < raygenPGs.size(); i++) {
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
		rec.data = nullptr;//暂时
		raygenRecords.push_back(rec);
	}
	raygenRecordsBuffer.alloc_and_upload(raygenRecords);
	sbt.raygenRecord = raygenRecordsBuffer.d_pointer();
	//----------------------------
	//创建未命中记录
	//------------------------------
	
	std::vector<MissRecord> missRecords;
	for (int i = 0; i < missPGs.size(); i++) {
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
		rec.data = nullptr;//暂时
		missRecords.push_back(rec);
	}
	missRecordsBuffer.alloc_and_upload(missRecords);
	sbt.missRecordBase = missRecordsBuffer.d_pointer();
	sbt.missRecordStrideInBytes = sizeof(MissRecord);
	sbt.missRecordCount = (int)missRecords.size();

	//---------------------------------------------
	//创建命中记录
	//---------------------------------------------
	//在实例中无物体，但是为了运行正常添加一个假的
	int numObjects = 1;
	std::vector<HitgroupRecord> hitgroupRecords;
	for (int i = 0; i < numObjects; i++) {
		int objectType = 0;
		HitgroupRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType], &rec));
		rec.objectID = i;
		hitgroupRecords.push_back(rec);
	}
	hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
	sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
	sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	sbt.hitgroupRecordCount = hitgroupRecords.size();
}
//构建BVH树
OptixTraversableHandle SampleRenderer::buildAccel(const TriangleMesh& model)
{
	//给设备上传模型
	vertexBuffer.alloc_and_upload(model.vertex);
	indexBuffer.alloc_and_upload(model.index);

	OptixTraversableHandle asHandle{ 0 };
	//-------------------------------------------
	//输入三角形
	//-------------------------------------------
	OptixBuildInput triangleInput = {};
	triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

	// create local variables, because we need a *pointer* to the
    // device pointers
	CUdeviceptr d_vertices = vertexBuffer.d_pointer();
	CUdeviceptr d_indices = indexBuffer.d_pointer();

	triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangleInput.triangleArray.vertexStrideInBytes = sizeof(gdt::vec3f);
	triangleInput.triangleArray.numVertices = (int)model.vertex.size();
	triangleInput.triangleArray.vertexBuffers = &d_vertices;

	triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	triangleInput.triangleArray.indexStrideInBytes = sizeof(gdt::vec3i);
	triangleInput.triangleArray.numIndexTriplets = (int)model.index.size();
	triangleInput.triangleArray.indexBuffer = d_indices;

	uint32_t triangleInputFlags[1] = { 0 };
	//我们有一个SBT入口，但是没有材质
	triangleInput.triangleArray.flags = triangleInputFlags;
	triangleInput.triangleArray.numSbtRecords = 1;
	triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
	triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
	triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

	//-----------------------------------------------
	//设置BLAS
	//---------------------------------------------
	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accelOptions.motionOptions.numKeys = 1;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blasBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelOptions, &triangleInput, 1, &blasBufferSizes));
	//-----------------------------------
	//prepare compaction(打包)
	//-------------------------------------
	CUDABuffer compactedSizeBuffer;
	compactedSizeBuffer.alloc(sizeof(uint64_t));

	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactedSizeBuffer.d_pointer();

	//-------------------------------
	//执行主步骤
	//------------------------------
	CUDABuffer tempBuffer;
	tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

	CUDABuffer outputBuffer;
	outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

	OPTIX_CHECK(optixAccelBuild(optixContext, 0/*stream*/, &accelOptions, &triangleInput,
		1, tempBuffer.d_pointer(), tempBuffer.sizeInBytes, outputBuffer.d_pointer(), outputBuffer.sizeInBytes, &asHandle, &emitDesc, 1));
	CUDA_SYNC_CHECK();

	//-----------------------------------------------------
	//perform 打包
	//---------------------------------------------
	uint64_t compactedSize;
	compactedSizeBuffer.download(&compactedSize, 1);

	asBuffer.alloc(compactedSize);
	OPTIX_CHECK(optixAccelCompact(optixContext, 0, asHandle, asBuffer.d_pointer(), asBuffer.sizeInBytes, &asHandle));

	CUDA_SYNC_CHECK();
	//------------------------------------------------------------------------
	//释放
	//----------------------------------------
	outputBuffer.free();
	tempBuffer.free();
	compactedSizeBuffer.free();

	return asHandle;
}

void TriangleMesh::addUnitCube(const gdt::affine3f& xfm)
{
	int firstVertexID = (int)vertex.size();
	vertex.push_back(gdt::xfmPoint(xfm, gdt::vec3f(0.f, 0.f, 0.f)));
	vertex.push_back(gdt::xfmPoint(xfm, gdt::vec3f(1.f, 0.f, 0.f)));
	vertex.push_back(gdt::xfmPoint(xfm, gdt::vec3f(0.f, 1.f, 0.f)));
	vertex.push_back(gdt::xfmPoint(xfm, gdt::vec3f(1.f, 1.f, 0.f)));
	vertex.push_back(gdt::xfmPoint(xfm, gdt::vec3f(0.f, 0.f, 1.f)));
	vertex.push_back(gdt::xfmPoint(xfm, gdt::vec3f(1.f, 0.f, 1.f)));
	vertex.push_back(gdt::xfmPoint(xfm, gdt::vec3f(0.f, 1.f, 1.f)));
	vertex.push_back(gdt::xfmPoint(xfm, gdt::vec3f(1.f, 1.f, 1.f)));

	int indices[] = { 0,1,3, 2,0,3,
				 5,7,6, 5,6,4,
				 0,4,5, 0,5,1,
				 2,3,7, 2,7,6,
				 1,5,7, 1,7,3,
				 4,0,2, 4,2,6
	};
	for (int i = 0; i < 12; i++) {
		index.push_back(firstVertexID + gdt::vec3i(indices[3 * i + 0], indices[3 * i + 1], indices[3 * i + 2]));
	}
}

void TriangleMesh::addCube(const gdt::vec3f& center, const gdt::vec3f& size)
{
	gdt::affine3f xfm;
	xfm.p = center - 0.5f * size;
	xfm.l.vx = gdt::vec3f(size.x, 0.f, 0.f);
	xfm.l.vy = gdt::vec3f(0.f, size.y, 0.f);
	xfm.l.vz = gdt::vec3f(0.f, 0.f, size.z);
	addUnitCube(xfm);
}
