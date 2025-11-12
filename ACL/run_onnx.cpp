#include <armnn/IRuntime.hpp>
#include <armnnOnnxParser/IOnnxParser.hpp>
#include <armnn/Logging.hpp>
//#include <armnn/LoggingService.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <memory>
#include <cstdint>


const unsigned int batchSize = 32;
const unsigned int channels  = 1;
const unsigned int height    = 28;
const unsigned int width     = 28;

// Utility: read big-endian integer from MNIST files
int32_t readInt(std::ifstream& f)
{
    unsigned char bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

// Load one MNIST image (28x28 = 784 bytes) normalized to [0,1]
std::vector<float> loadMnistImage(std::ifstream& f, int index, int rows = 28, int cols = 28)
{
    f.seekg(16 + index * rows * cols);  // skip header + previous images
    std::vector<unsigned char> buffer(rows * cols);
    f.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    std::vector<float> img(buffer.size());
    for (size_t i = 0; i < buffer.size(); i++)
        img[i] = buffer[i] / 255.0f;   // normalize
    return img;
}

// Load one MNIST label
int loadMnistLabel(std::ifstream& f, int index)
{
    f.seekg(8 + index); // skip header + previous labels
    unsigned char label;
    f.read(reinterpret_cast<char*>(&label), 1);
    return label;
}

int main()
{
    try
    {
        // Paths to dataset files
        std::string imageFile = "train-images-idx3-ubyte";
        std::string labelFile = "train-labels-idx1-ubyte";
        std::string modelFile = "mnist_cnn.onnx";

        // Open dataset files
        std::ifstream imgStream(imageFile, std::ios::binary);
        std::ifstream lblStream(labelFile, std::ios::binary);

        if (!imgStream.is_open() || !lblStream.is_open())
        {
            std::cerr << "Failed to open MNIST dataset files!" << std::endl;
            return -1;
        }

        // Read headers
        int magicImages = readInt(imgStream);
        int numImages = readInt(imgStream);
        int rows = readInt(imgStream);
        int cols = readInt(imgStream);

        int magicLabels = readInt(lblStream);
        int numLabels = readInt(lblStream);

        std::cout << "Dataset contains " << numImages << " images, " << rows << "x" << cols << std::endl;

        // Select one image (e.g., index 0)
        int index = 5;
        std::vector<float> inputData = loadMnistImage(imgStream, index, rows, cols);
        int trueLabel = loadMnistLabel(lblStream, index);

        // 1. Parse ONNX model
        //armnn::ConfigureLogging(/*printToStdOut*/ true,/*printToDebug*/ true,armnn::LogSeverity::off);
        auto parser = armnnOnnxParser::IOnnxParser::Create();
        auto network = parser->CreateNetworkFromBinaryFile("mnist_cnn.onnx");

        // 2. Create runtime
        armnn::IRuntime::CreationOptions options;
        armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

        // Optimize
        //armnn::OptimizerOptions options;
        //options.m_ReduceFp32ToFp16 = true;
        armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(
            *network,
            {armnn::Compute:: CpuRef},  // try CpuAcc (NEON); fallback CpuRef
            runtime->GetDeviceSpec()
        );

        armnn::NetworkId networkId;
        runtime->LoadNetwork(networkId, std::move(optNet));

        // 3. Setup input tensor
		//auto inputInfo = runtime->GetInputTensorInfo(networkId, 0);
		armnn::TensorInfo inputInfo({batchSize, channels, height, width}, armnn::DataType::Float32);
		inputInfo.SetConstant(true);
		std::cout << "Model expects input shape: ";
		for (unsigned int i = 0; i < inputInfo.GetNumDimensions(); ++i)
			std::cout << inputInfo.GetShape()[i] << " ";
		std::cout << std::endl;

		std::cout << "Model expects " << inputInfo.GetNumElements() << " elements" << std::endl;
		std::cout << "Input data has " << inputData.size() << " elements" << std::endl;

		std::vector<float> batchedInput;
		batchedInput.reserve(32 * 784);

		for (int i = 0; i < 32; ++i)
			batchedInput.insert(batchedInput.end(), inputData.begin(), inputData.end());

		std::cout << "Batched input data has " << batchedInput.size() << " elements" << std::endl;

		// Create Tensor instead of ConstTensor
		//std::vector<float> inputImage(inputInfo.GetNumElements(), 0.0f);
		armnn::ConstTensor inputTensor(inputInfo, batchedInput.data());
		armnn::InputTensors inputTensors{{0, inputTensor}};

        //armnn::InputTensors inputTensors{
            //{0, armnn::ConstTensor(runtime->GetInputTensorInfo(networkId, 0), inputData.data())}
       // };

        // 4. Setup output tensor (10 classes)
        std::vector<float> outputData(10);
        armnn::OutputTensors outputTensors{
            {0, armnn::Tensor(runtime->GetOutputTensorInfo(networkId, 0), outputData.data())}
        };

        // 5. Run inference
        runtime->EnqueueWorkload(networkId, inputTensors, outputTensors);

        // 6. Get prediction
        int predictedDigit = std::distance(
            outputData.begin(),
            std::max_element(outputData.begin(), outputData.end())
        );

        std::cout << "True label: " << trueLabel << " | Predicted: " << predictedDigit << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
