#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>

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

// stuff we know about the network and the caffe input/output blobs

static const int BATCH_SIZE = 4;
static const int TIMING_ITERATIONS = 1000;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";


// Logger for GIE info/warning/errors
class Logger : public ILogger			
{
	void log(Severity severity, const char* msg) override
	{
		if (severity!=Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;


std::string locateFile(const std::string& input)
{
	std::string file = "data/samples/googlenet/" + input;
	struct stat info;
	int i, MAX_DEPTH = 10;
	for (i = 0; i < MAX_DEPTH && stat(file.c_str(), &info); i++)
		file = "../" + file;

	assert(i != MAX_DEPTH);

	return file;
}

struct Profiler : public IProfiler
{
	typedef std::pair<std::string, float> Record;
	std::vector<Record> mProfile;

	virtual void reportLayerTime(const char* layerName, float ms)
	{
		auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
		if (record == mProfile.end())
			mProfile.push_back(std::make_pair(layerName, ms));
		else
			record->second += ms;
	}

	void printLayerTimes()
	{
		float totalTime = 0;
		for (size_t i = 0; i < mProfile.size(); i++)
		{
			printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);
			totalTime += mProfile[i].second;
		}
		printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
	}

} gProfiler;

void caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
					 const std::string& modelFile,				// name for model 
					 const std::vector<std::string>& outputs,   // network outputs
					 unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
					 std::ostream& gieModelStream)
{
	// create API root class - must span the lifetime of the engine usage
	IBuilder* builder = createInferBuilder(gLogger);
	INetworkDefinition* network = builder->createNetwork();

	// parse the caffe model to populate the network, then set the outputs
	ICaffeParser* parser = createCaffeParser();

	bool useFp16 = builder->platformHasFastFp16();

	DataType modelDataType = useFp16 ? DataType::kHALF : DataType::kFLOAT; // create a 16-bit model if it's natively supported
	const IBlobNameToTensor *blobNameToTensor =
		parser->parse(locateFile(deployFile).c_str(),				// caffe deploy file
								 locateFile(modelFile).c_str(),		// caffe model file
								 *network,							// network definition that the parser will populate
								 modelDataType);

	assert(blobNameToTensor != nullptr);
	// the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate	
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(16 << 20);

	// set up the network for paired-fp16 format if available
	if(useFp16)
		builder->setHalf2Mode(true);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	engine->serialize(gieModelStream);
	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

void timeInference(ICudaEngine* engine, int batchSize)
{
	// input and output buffer pointers that we pass to the engine - the engine requires exactly ICudaEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine->getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than ICudaEngine::getNbBindings()
	int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME), outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

	// allocate GPU buffers
	Dims3 inputDims = engine->getBindingDimensions(inputIndex), outputDims = engine->getBindingDimensions(outputIndex);
	size_t inputSize = batchSize * inputDims.c * inputDims.h * inputDims.w * sizeof(float);
	size_t outputSize = batchSize * outputDims.c * outputDims.h * outputDims.w * sizeof(float);

	CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
	CHECK(cudaMalloc(&buffers[outputIndex], outputSize));

	IExecutionContext* context = engine->createExecutionContext();
	context->setProfiler(&gProfiler);

	// zero the input buffer
	CHECK(cudaMemset(buffers[inputIndex], 0, inputSize));

	for (int i = 0; i < TIMING_ITERATIONS;i++)
		context->execute(batchSize, buffers);

	// release the context and buffers
	context->destroy();
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}


int main(int argc, char** argv)
{
	std::cout << "Building and running a GPU inference engine for GoogleNet, N=4..." << std::endl;

	// parse the caffe model and the mean file
	std::stringstream gieModelStream;
	gieModelStream.seekg(0, gieModelStream.beg);
	caffeToGIEModel("googlenet.prototxt", "googlenet.caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, BATCH_SIZE, gieModelStream);

	// create an engine
	IRuntime* infer = createInferRuntime(gLogger);
	ICudaEngine* engine = infer->deserializeCudaEngine(gieModelStream);

	// run inference with null data to time network performance
	timeInference(engine, BATCH_SIZE);

	engine->destroy();
	infer->destroy();

	gProfiler.printLayerTimes();

	std::cout << "Done." << std::endl;

	return 0;
}
