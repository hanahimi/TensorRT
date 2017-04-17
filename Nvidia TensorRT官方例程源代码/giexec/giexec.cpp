#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <string.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}

struct Params
{
	std::string deployFile, modelFile, engine;
	std::vector<std::string> outputs;
	int device{ 0 }, batchSize{ 1 }, workspaceSize{ 16 }, iterations{ 10 }, avgRuns{ 10 };
	bool half2{ false }, verbose{ false }, hostTime{ false };
} gParams;

std::vector<std::string> gInputs;

// Logger for GIE info/warning/errors
class Logger : public ILogger			
{
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO || gParams.verbose)
			std::cout << msg << std::endl;
	}
} gLogger;


ICudaEngine* caffeToGIEModel()
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	const IBlobNameToTensor* blobNameToTensor = parser->parse(gParams.deployFile.c_str(),
															  gParams.modelFile.c_str(),
															  *network,
															  gParams.half2 ? DataType::kHALF:DataType::kFLOAT);


	if (!blobNameToTensor)
		return nullptr;

	for (int i = 0, n = network->getNbInputs(); i < n; i++)
		gInputs.push_back(network->getInput(i)->getName());

	// specify which tensors are outputs
	for (auto& s : gParams.outputs)
	{
		if (blobNameToTensor->find(s.c_str()) == nullptr)
		{
			std::cout << "could not find output blob " << s << std::endl;
			return nullptr;
		}
		network->markOutput(*blobNameToTensor->find(s.c_str()));
	}

	// Build the engine
	builder->setMaxBatchSize(gParams.batchSize);
	builder->setMaxWorkspaceSize(gParams.workspaceSize<<20);
	builder->setHalf2Mode(gParams.half2);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	if (engine == nullptr)
		std::cout << "could not build engine" << std::endl;

	parser->destroy();
	network->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
	return engine;
}

void createMemory(const ICudaEngine& engine, std::vector<void*>& buffers, const std::string& name)
{
	size_t bindingIndex = engine.getBindingIndex(name.c_str());
	assert(bindingIndex < buffers.size());
	Dims3 dimensions = engine.getBindingDimensions((int)bindingIndex);
	size_t eltCount = dimensions.c*dimensions.h*dimensions.w*gParams.batchSize, memSize = eltCount * sizeof(float);

	float* localMem = new float[eltCount];
	for (size_t i = 0; i < eltCount; i++)
		localMem[i] = (float(rand()) / RAND_MAX) * 2 - 1;

	void* deviceMem;
	CHECK(cudaMalloc(&deviceMem, memSize));
	if (deviceMem == nullptr)
	{
		std::cerr << "Out of memory" << std::endl;
		exit(1);
	}
	CHECK(cudaMemcpy(deviceMem, localMem, memSize, cudaMemcpyHostToDevice));

	delete[] localMem;
	buffers[bindingIndex] = deviceMem;	
}

void doInference(ICudaEngine& engine)
{
	IExecutionContext *context = engine.createExecutionContext();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.

	std::vector<void*> buffers(gInputs.size() + gParams.outputs.size());
	for (size_t i = 0; i < gInputs.size(); i++)
		createMemory(engine, buffers, gInputs[i]);

	for (size_t i = 0; i < gParams.outputs.size(); i++)
		createMemory(engine, buffers, gParams.outputs[i]);

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	cudaEvent_t start, end;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&end));

	for (int j = 0; j < gParams.iterations; j++)
	{
		float total = 0, ms;
		for (int i = 0; i < gParams.avgRuns; i++)
		{
			if (gParams.hostTime)
			{
				auto t_start = std::chrono::high_resolution_clock::now();
				context->execute(gParams.batchSize, &buffers[0]);
				auto t_end = std::chrono::high_resolution_clock::now();
				ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
			}
			else
			{
				cudaEventRecord(start, stream);
				context->enqueue(gParams.batchSize, &buffers[0], stream, nullptr);
				cudaEventRecord(end, stream);
				cudaEventSynchronize(end);
				cudaEventElapsedTime(&ms, start, end);
			}
			total += ms;
		}
		total /= gParams.avgRuns;
		std::cout << "Average over " << gParams.avgRuns << " runs is " << total << " ms." << std::endl;
	}


	cudaStreamDestroy(stream);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}



static void printUsage()
{
	printf("\n");
	printf("Mandatory params:\n");
	printf("  --model=<file>       Caffe model file\n");
	printf("  --deploy=<file>      Caffe deploy file\n");
	printf("  --output=<name>      Output blob name (can be specified multiple times\n");

	printf("\nOptional params:\n");

	printf("  --batch=N            Set batch size (default = %d)\n", gParams.batchSize);
	printf("  --device=N           Set cuda device to N (default = %d)\n", gParams.device);
	printf("  --iterations=N       Run N iterations (default = %d)\n", gParams.iterations);
	printf("  --avgRuns=N          Set avgRuns to N - perf is measured as an average of avgRuns (default=%d)\n", gParams.avgRuns);
	printf("  --workspace=N        Set workspace size in megabytes (default = %d)\n", gParams.workspaceSize);
	printf("  --half2              Run in paired fp16 mode - default = false\n");
	printf("  --verbose            Use verbose logging - default = false\n");
	printf("  --hostTime	       Measure host time rather than GPU time - default = false\n");
	printf("  --engine=<file>      Generate a serialized GIE engine\n");

	fflush(stdout);
}

bool parseString(const char* arg, const char* name, std::string& value)
{
	size_t n = strlen(name);
	bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
	if (match)
	{
		value = arg + n + 3;
		std::cout << name << ": " << value << std::endl;
	}
	return match;
}

bool parseInt(const char* arg, const char* name, int& value)
{
	size_t n = strlen(name);
	bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
	if (match)
	{
		value = atoi(arg + n + 3);
		std::cout << name << ": " << value << std::endl;
	}
	return match;
}

bool parseBool(const char* arg, const char* name, bool& value)
{
	size_t n = strlen(name);
	bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n);
	if (match)
	{
		std::cout << name << std::endl;
		value = true;
	}
	return match;

}


bool parseArgs(int argc, char* argv[])
{
	if (argc < 4)
	{
		printUsage();
		return false;
	}

	for (int j = 1; j < argc; j++)
	{
		if (parseString(argv[j], "model", gParams.modelFile) || parseString(argv[j], "deploy", gParams.deployFile) || parseString(argv[j], "engine", gParams.engine))
			continue;
		
		std::string output;
		if (parseString(argv[j], "output", output))
		{
			gParams.outputs.push_back(output);
			continue;
		}

		if (parseInt(argv[j], "batch", gParams.batchSize) || parseInt(argv[j], "iterations", gParams.iterations) || parseInt(argv[j], "avgRuns", gParams.avgRuns) 
			|| parseInt(argv[j], "device", gParams.device)	|| parseInt(argv[j], "workspace", gParams.workspaceSize))
			continue;

		if (parseBool(argv[j], "half2", gParams.half2) || parseBool(argv[j], "verbose", gParams.verbose) || parseBool(argv[j], "hostTime", gParams.hostTime))
			continue;

		printf("Unknown argument: %s\n", argv[j]);
		return false;
	}
	return true;
}

int main(int argc, char** argv)
{
	// create a GIE model from the caffe model and serialize it to a stream

	if (!parseArgs(argc, argv))
		return -1;

	cudaSetDevice(gParams.device);

	if (gParams.modelFile.empty() || gParams.deployFile.empty())
	{
		std::cerr << "Model or deploy file not specified" << std::endl;
		return -1;
	}

	if (gParams.outputs.size() == 0)
	{
		std::cerr << "At least one network output must be defined" << std::endl;
		return -1;
	}

	ICudaEngine* engine = caffeToGIEModel();
	if (!engine)
	{
		std::cerr << "Engine could not be created" << std::endl;
		return -1;
	}

	if (!gParams.engine.empty())
	{
		std::ofstream p(gParams.engine);
		if (!p)
		{
			std::cerr << "could not open plan output file" << std::endl;
			return -1;
		}
		engine->serialize(p);
	}

	doInference(*engine);
	engine->destroy();

	return 0;
}
