#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <map>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <cstring>
using namespace nvinfer1;
using namespace nvcaffeparser1;

// Our weight files are in a very simple space delimited format.
// [type] [size] <data x size in hex> 
static std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::map<std::string, Weights> weightMap;
	std::ifstream input(file);
	assert(input.is_open() && "Unable to load weight file.");
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while(count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t type, size;
        std::string name;
        input >> name >> std::dec >> type >> size;
        wt.type = static_cast<DataType>(type);
        if (wt.type == DataType::kFLOAT)
        {
            uint32_t *val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];

            }
            wt.values = val;
        } else if (wt.type == DataType::kHALF)
        {
            uint16_t *val = reinterpret_cast<uint16_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}
#define CHECK(status)					\
{							\
    if (status != 0)				\
    {						\
        std::cout << "Cuda failure: " << status;\
		abort();				\
	}						\
}


// stuff we know about the network and the caffe input/output blobs
static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE = 10;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

// Logger for GIE info/warning/errors
class Logger : public ILogger			
{
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;


std::string locateFile(const std::string& input)
{
	std::string file = "data/samples/mnist/" + input;
	struct stat info;
	int i, MAX_DEPTH = 10;
	for (i = 0; i < MAX_DEPTH && stat(file.c_str(), &info); i++)
		file = "../" + file;

	assert(i != MAX_DEPTH);

	return file;
}

// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& fileName,  uint8_t buffer[INPUT_H*INPUT_W])
{
	std::ifstream infile(locateFile(fileName), std::ifstream::binary);
	std::string magic, h, w, max;
	infile >> magic >> h >> w >> max;
	infile.seekg(1, infile.cur);
	infile.read(reinterpret_cast<char*>(buffer), INPUT_H*INPUT_W);
}
ICudaEngine *
createMNISTEngine(unsigned int maxBatchSize, IBuilder *builder, DataType dt)
{
	INetworkDefinition* network = builder->createNetwork();
//  input: "data"
//  input_shape {
//    dim: 1
//    dim: 1
//    dim: 28
//    dim: 28
//  }
//  Create input
	auto data = network->addInput("data", dt, Dims3{ 1, 28, 28});
	assert(data != nullptr);
//  layer {
//    name: "scale"
//    type: "Power"
//    bottom: "data"
//    top: "scale"
//    power_param {
//      scale: 0.0125
//    }
//  }
	float scale_param = 0.0125f;
	float shift_param = 0.0f;
	float power_param = 1.0;
	Weights power{DataType::kFLOAT, &power_param, 1};
	Weights shift{DataType::kFLOAT, &shift_param, 1};
	Weights scale{DataType::kFLOAT, &scale_param, 1};
	auto scale_1 = network->addScale(*data,	ScaleMode::kUNIFORM, shift, scale, power);
	assert(scale_1 != nullptr);
//  layer {
//    name: "conv1"
//    type: "Convolution"
//    bottom: "scale"
//    top: "conv1"
//    param {
//      lr_mult: 1.0
//    }
//    param {
//      lr_mult: 2.0
//    }
//    convolution_param {
//      num_output: 20
//      kernel_size: 5
//      stride: 1
//      weight_filler {
//        type: "xavier"
//      }
//      bias_filler {
//        type: "constant"
//      }
//    }
//  }
    std::map<std::string, Weights> &&weightMap = loadWeights(locateFile("mnistgie.wts"));
	auto conv_1 = network->addConvolution(*scale_1->getOutput(0), 20, Dims2{5, 5}, weightMap["conv1filter"], weightMap["conv1bias"]);
	assert(conv_1 != nullptr);
	conv_1->setStride(Dims2{1, 1});
//  layer {
//    name: "pool1"
//    type: "Pooling"
//    bottom: "conv1"
//    top: "pool1"
//    pooling_param {
//      pool: MAX
//      kernel_size: 2
//      stride: 2
//    }
//  }
	auto pool_1 = network->addPooling(*conv_1->getOutput(0), PoolingType::kMAX, Dims2{2, 2});
	assert(pool_1 != nullptr);
	pool_1->setStride(Dims2{2, 2});

//  layer {
//    name: "conv2"
//    type: "Convolution"
//    bottom: "pool1"
//    top: "conv2"
//    param {
//      lr_mult: 1.0
//    }
//    param {
//      lr_mult: 2.0
//    }
//    convolution_param {
//      num_output: 50
//      kernel_size: 5
//      stride: 1
//      weight_filler {
//        type: "xavier"
//      }
//      bias_filler {
//        type: "constant"
//      }
//    }
//  }
	auto conv_2 = network->addConvolution(*pool_1->getOutput(0), 50, Dims2{5, 5}, weightMap["conv2filter"], weightMap["conv2bias"]);
	assert(conv_2 != nullptr);
	conv_2->setStride(Dims2{1, 1});
//  layer {
//    name: "pool2"
//    type: "Pooling"
//    bottom: "conv2"
//    top: "pool2"
//    pooling_param {
//      pool: MAX
//      kernel_size: 2
//      stride: 2
//    }
//  }
	auto pool_2 = network->addPooling(*conv_2->getOutput(0), PoolingType::kMAX, Dims2{2, 2});
	assert(pool_2 != nullptr);
	pool_2->setStride(Dims2{2, 2});
//  layer {
//    name: "ip1"
//    type: "InnerProduct"
//    bottom: "pool2"
//    top: "ip1"
//    param {
//      lr_mult: 1.0
//    }
//    param {
//      lr_mult: 2.0
//    }
//    inner_product_param {
//      num_output: 500
//      weight_filler {
//        type: "xavier"
//      }
//      bias_filler {
//        type: "constant"
//      }
//    }
//  }
	auto ip_1 = network->addFullyConnected(*pool_2->getOutput(0), 500, weightMap["ip1filter"], weightMap["ip1bias"]);
	assert(ip_1 != nullptr);
//  layer {
//    name: "relu1"
//    type: "ReLU"
//    bottom: "relu1"
//    top: "ip1"
//  }
	auto relu_1 = network->addActivation(*ip_1->getOutput(0), ActivationType::kRELU);
	assert(relu_1 != nullptr);
//  layer {
//    name: "ip2"
//    type: "InnerProduct"
//    bottom: "relu1"
//    top: "ip2"
//    param {
//      lr_mult: 1.0
//    }
//    param {
//      lr_mult: 2.0
//    }
//    inner_product_param {
//      num_output: 10
//      weight_filler {
//        type: "xavier"
//      }
//      bias_filler {
//        type: "constant"
//      }
//    }
//  }
	auto ip_2 = network->addFullyConnected(*relu_1->getOutput(0), 10, weightMap["ip2filter"], weightMap["ip2bias"]);
	assert(ip_2 != nullptr);
//  layer {
//    name: "prob"
//    type: "Softmax"
//    bottom: "ip2"
//    top: "prob"
//  }
	auto prob = network->addSoftMax(*ip_2->getOutput(0));
	assert(prob != nullptr);
	prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
	network->markOutput(*prob->getOutput(0));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);

	auto engine = builder->buildCudaEngine(*network);
	// we don't need the network any more
	network->destroy();

	// Once we have built the cuda engine, we can release all of our held memory.
	for (auto &mem : weightMap)
    {
        free((void*)(mem.second.values));
    }
	return engine;
}

void APIToGIEModel(unsigned int maxBatchSize, // batch size - NB must be at least as large as the batch we want to run with)
		     std::ostream& gieModelStream) // output stream for the GIE model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs and create an engine
	ICudaEngine* engine = createMNISTEngine(maxBatchSize, builder, DataType::kFLOAT);

	assert(engine != nullptr);

	// serialize the engine, then close everything down
	engine->serialize(gieModelStream);
	engine->destroy();
	builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME), 
	    outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
	// create a GIE model from the caffe model and serialize it to a stream
	std::stringstream gieModelStream;
	APIToGIEModel(1, gieModelStream);

	// read a random digit file
	srand(unsigned(time(nullptr)));
	uint8_t fileData[INPUT_H*INPUT_W];
	readPGMFile(std::to_string(rand() % 10) + ".pgm", fileData);

	// print an ascii representation
	std::cout << "\n\n\n---------------------------" << "\n\n\n" << std::endl;
	for (int i = 0; i < INPUT_H*INPUT_W; i++)
		std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");

	// parse the mean file and 	subtract it from the image
	ICaffeParser* parser = createCaffeParser();
	IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(locateFile("mnist_mean.binaryproto").c_str());
	parser->destroy();
	const float *meanData = reinterpret_cast<const float*>(meanBlob->getData());

	float data[INPUT_H*INPUT_W];
	for (int i = 0; i < INPUT_H*INPUT_W; i++)
		data[i] = float(fileData[i])-meanData[i];

	meanBlob->destroy();

	// deserialize the engine 
	gieModelStream.seekg(0, gieModelStream.beg);

	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream);

	IExecutionContext *context = engine->createExecutionContext();

	// run inference
	float prob[OUTPUT_SIZE];
	doInference(*context, data, prob, 1);

	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	// print a histogram of the output distribution
	std::cout << "\n\n";
	for (unsigned int i = 0; i < 10; i++)
		std::cout << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
	std::cout << std::endl;

	return 0;
}
