#include <armnn/IRuntime.hpp>
#include <armnnOnnxParser/IOnnxParser.hpp>
#include <armnn/Logging.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <memory>
#include <cstdint>
#include <chrono>

// ========================
// Configuration
// ========================
constexpr unsigned int batchSize = 1;
constexpr unsigned int channels  = 1;
constexpr unsigned int height    = 28;
constexpr unsigned int width     = 28;

constexpr unsigned int numClasses = 10;
constexpr unsigned int NUM_SAMPLES = 5032;
constexpr unsigned int WARMUP_RUNS = 2;

// ========================
// MNIST utilities
// ========================
int32_t readInt(std::ifstream& f)
{
    unsigned char bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) |
           (bytes[1] << 16) |
           (bytes[2] << 8)  |
           bytes[3];
}

std::vector<float> loadMnistImage(std::ifstream& f, int index, int rows, int cols)
{
    f.seekg(16 + index * rows * cols);
    std::vector<unsigned char> buffer(rows * cols);
    f.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    std::vector<float> img(buffer.size());
    for (size_t i = 0; i < buffer.size(); ++i)
    {
        float val = buffer[i] / 255.0f;
        img[i] = (val - 0.1307f) / 0.3081f;
    }
    return img;
}

int loadMnistLabel(std::ifstream& f, int index)
{
    f.seekg(8 + index);
    unsigned char label;
    f.read(reinterpret_cast<char*>(&label), 1);
    return static_cast<int>(label);
}

// ========================
// Main
// ========================
int main()
{
    try
    {
        // ------------------------
        // Files
        // ------------------------
        const std::string imageFile = "t10k-images-idx3-ubyte";
        const std::string labelFile = "t10k-labels-idx1-ubyte";
        const std::string modelFile = "/home/orangepi/Documents/Project/models/mnist_cnn_b1.onnx";

        std::ifstream imgStream(imageFile, std::ios::binary);
        std::ifstream lblStream(labelFile, std::ios::binary);

        if (!imgStream || !lblStream)
        {
            std::cerr << "Failed to open MNIST files\n";
            return -1;
        }

        // ------------------------
        // Read MNIST headers
        // ------------------------
        readInt(imgStream);                 // magic
        int totalImages = readInt(imgStream);
        int rows = readInt(imgStream);
        int cols = readInt(imgStream);

        readInt(lblStream);                 // magic
        readInt(lblStream);                 // num labels

        if (totalImages < NUM_SAMPLES)
        {
            std::cerr << "Not enough MNIST samples\n";
            return -1;
        }

        std::cout << "Loaded MNIST: " << rows << "x" << cols << "\n";

        // ------------------------
        // Preload MNIST (NO TIMING)
        // ------------------------
        std::vector<std::vector<float>> images(NUM_SAMPLES);
        std::vector<int> labels(NUM_SAMPLES);

        for (unsigned int i = 0; i < NUM_SAMPLES; ++i)
        {
            images[i] = loadMnistImage(imgStream, i, rows, cols);
            labels[i] = loadMnistLabel(lblStream, i);
        }

        // ------------------------
        // Load ONNX model
        // ------------------------
        auto parser = armnnOnnxParser::IOnnxParser::Create();
        auto network = parser->CreateNetworkFromBinaryFile(modelFile.c_str());

        armnn::IRuntime::CreationOptions options;
        armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));


        auto optNet = armnn::Optimize(
            *network,
            { armnn::Compute::CpuRef },
            runtime->GetDeviceSpec()
        );

        armnn::NetworkId networkId;
        runtime->LoadNetwork(networkId, std::move(optNet));

        // ------------------------
        // Tensor setup
        // ------------------------
        armnn::TensorInfo inputInfo(
            { batchSize, channels, height, width },
            armnn::DataType::Float32
        );
        inputInfo.SetConstant(true);


        armnn::TensorInfo outputInfo(
            { batchSize, numClasses },
            armnn::DataType::Float32
        );
        outputInfo.SetConstant(true);

        std::vector<float> batchedInput(batchSize * height * width);
        std::vector<float> outputData(batchSize * numClasses);

        armnn::Tensor inputTensor(inputInfo, batchedInput.data());
        armnn::Tensor outputTensor(outputInfo, outputData.data());

        armnn::InputTensors inputTensors{ {0, inputTensor} };
        armnn::OutputTensors outputTensors{ {0, outputTensor} };

        // ------------------------
        // Benchmark loop
        // ------------------------
        using clock = std::chrono::high_resolution_clock;

        double totalInferenceMs = 0.0;
        unsigned int inferenceRuns = 0;

        int correct = 0;
        int total = 0;

        for (unsigned int idx = 0;
             idx + batchSize <= NUM_SAMPLES;
             idx += batchSize)
        {
            // Fill batch
            for (unsigned int b = 0; b < batchSize; ++b)
            {
                std::copy(
                    images[idx + b].begin(),
                    images[idx + b].end(),
                    batchedInput.begin() + b * height * width
                );
            }

            // Warm-up
            if (inferenceRuns < WARMUP_RUNS)
            {
                std::cout<<"warmup dataset \n";
                runtime->EnqueueWorkload(
                    networkId, inputTensors, outputTensors);
                inferenceRuns++;
                continue;
            
            }

            // Timed inference
            auto t0 = clock::now();
            runtime->EnqueueWorkload(
                networkId, inputTensors, outputTensors);
            auto t1 = clock::now();

            totalInferenceMs +=
                std::chrono::duration<double, std::milli>(t1 - t0).count();

            inferenceRuns++;

            // Accuracy
            for (unsigned int b = 0; b < batchSize; ++b)
            {
                float* row = outputData.data() + b * numClasses;
                int pred = std::max_element(row, row + numClasses) - row;
                
                if (pred == labels[idx + b]){
                    correct++;}
                else{
                    std::cout<<"predicted value"<<pred<<"  "<<"label value"<<labels[idx + b]<<std::endl;}

                total++;
            }
        }

        // ------------------------
        // Results
        // ------------------------
        unsigned int measuredRuns = inferenceRuns - WARMUP_RUNS;

        std::cout << "\n==== BENCHMARK RESULTS ====\n";
        std::cout << "Batch size              : " << batchSize << "\n";
        std::cout << "Images processed        : " << total << "\n";
        std::cout << "Avg batch inference time: "
                  << (totalInferenceMs / measuredRuns) << " ms\n";
        std::cout << "Per-image latency       : "
                  << (totalInferenceMs / measuredRuns) / batchSize << " ms\n";
        std::cout << "Accuracy                : "
                  << (100.0 * correct / total) << " %\n";

    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
