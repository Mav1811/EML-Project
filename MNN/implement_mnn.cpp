#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"

using namespace MNN;
using namespace std;

// ============================
// Helper functions
// ============================

std::vector<uint8_t> readIdx3(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open " + filename);

    int32_t magic = 0, num = 0, rows = 0, cols = 0;
    file.read((char*)&magic, 4);
    file.read((char*)&num, 4);
    file.read((char*)&rows, 4);
    file.read((char*)&cols, 4);

    // Convert from big endian
    magic = __builtin_bswap32(magic);
    num = __builtin_bswap32(num);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    std::vector<uint8_t> data(num * rows * cols);
    file.read((char*)data.data(), data.size());
    return data;
}

std::vector<uint8_t> readIdx1(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open " + filename);

    int32_t magic = 0, num = 0;
    file.read((char*)&magic, 4);
    file.read((char*)&num, 4);

    magic = __builtin_bswap32(magic);
    num = __builtin_bswap32(num);

    std::vector<uint8_t> labels(num);
    file.read((char*)labels.data(), labels.size());
    return labels;
}

// ============================
// Main
// ============================

int main() {
    const std::string model_path = "/home/orangepi/Documents/Project/MNN/mnist_cnn.mnn";
    const std::string image_path = "/home/orangepi/Documents/Project/MNN/t10k-images-idx3-ubyte";
    const std::string label_path = "/home/orangepi/Documents/Project/MNN/t10k-labels-idx1-ubyte";

    // Load MNIST data
    auto images = readIdx3(image_path);
    auto labels = readIdx1(label_path);

    // Create MNN interpreter and session
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(model_path.c_str()));

    ScheduleConfig config;
    config.type = MNN_FORWARD_OPENCL;
    config.numThread = 1; // adjust if needed
	//config.openCLRuntimeDebug = true;

    BackendConfig backendConfig;
    backendConfig.precision = BackendConfig::Precision_Normal;
    config.backendConfig = &backendConfig;
   

    auto session = net->createSession(config);
    auto inputTensor = net->getSessionInput(session, nullptr);

    // Assume input is [1,1,28,28]
    const int img_h = 28, img_w = 28;

    // Host tensor for input in NCHW
    Tensor inputHost(inputTensor, inputTensor->getDimensionType());

    int correct = 0;
    int num_tests = 10; // test 10 samples

    for (int i = 0; i < num_tests; ++i) {
        // Normalize image (0-1 range)
        for (int j = 0; j < img_h * img_w; ++j) {
            float pixel = images[i * img_h * img_w + j] / 255.0f;
            pixel =  (pixel - 0.1307f)/0.3081f;
            inputHost.host<float>()[j] = pixel;
            
        }

        // Copy data to device (MNN handles NCHW→NC4HW4)
        inputTensor->copyFromHostTensor(&inputHost);

        auto start = std::chrono::high_resolution_clock::now();
        net->runSession(session);
        auto end = std::chrono::high_resolution_clock::now();
        double inference_time =
            std::chrono::duration<double, std::milli>(end - start).count();

        // Get output
        auto outputTensor = net->getSessionOutput(session, nullptr);
        Tensor outputHost(outputTensor, outputTensor->getDimensionType());
        outputTensor->copyToHostTensor(&outputHost);

        // Find predicted label (argmax)
        float max_val = -1e9f;
        int predicted = -1;
        /*cout<<"The RAW values of the output";*/ 
        for (int k = 0; k < outputHost.elementSize(); ++k) {
            float val = outputHost.host<float>()[k];
            /*cout<< val;*/
            if (val > max_val) {
                max_val = val;
                predicted = k;
            }
        }

        int true_label = labels[i];
        if (predicted == true_label) correct++;

        std::cout << "Sample " << i
                  << ": Predicted=" << predicted
                  << ", True=" << true_label
                  << ", Time=" << inference_time << " ms"
                  << std::endl;
    }

    std::cout << "Accuracy on first " << num_tests << " samples: "
              << (float)correct / num_tests * 100.0f << "%" << std::endl;
        

    return 0;
}
