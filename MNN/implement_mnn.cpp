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
    const std::string model_path_32 = "/home/orangepi/Documents/Project/models/mnist_cnn_b32.mnn";
    const std::string model_path_16 = "/home/orangepi/Documents/Project/models/mnist_cnn_b16.mnn";
    const std::string model_path_8 = "/home/orangepi/Documents/Project/models/mnist_cnn_b8.mnn";
    const std::string model_path_1 = "/home/orangepi/Documents/Project/models/mnist_cnn_b1.mnn";
    const std::string image_path = "/home/orangepi/Documents/data/MNIST/raw/t10k-images-idx3-ubyte";
    const std::string label_path = "/home/orangepi/Documents/data/MNIST/raw/t10k-labels-idx1-ubyte";

    // Load MNIST data
    auto images = readIdx3(image_path);
    auto labels = readIdx1(label_path);

    // Create MNN interpreter and session
   
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(model_path_1.c_str()));    

    ScheduleConfig config;
    config.type = MNN_FORWARD_OPENCL;
    config.numThread = 1; // adjust if needed
	//config.openCLRuntimeDebug = true;

    BackendConfig backendConfig;
    backendConfig.precision = BackendConfig::Precision_Normal;
    config.backendConfig = &backendConfig;
   

    auto session = net->createSession(config);
    auto inputTensor = net->getSessionInput(session, nullptr);
    auto inputshape = inputTensor->shape();



    // Host tensor for input in NCHW
    Tensor inputHost(inputTensor, inputTensor->getDimensionType());
    cout<<"the input tesor dimensions"<<inputTensor->getDimensionType()<<endl;
    cout<<" The input tensor size"<<inputTensor->size()<<endl;
    cout<<" The input tensor element size"<<inputTensor->elementSize()<<endl;
    cout << "Input tensor shape: [ ";
    for (int d : inputshape) cout << d << " ";
    cout << "]" << endl;
    int batch_size = inputshape[0];     
    int channels   = inputshape[1];     // 1
    int height     = inputshape[2];     // 28
    int width      = inputshape[3];     // 28

    //output tensor definition
    auto outputTensor = net->getSessionOutput(session, nullptr);
    Tensor outputHost(outputTensor, outputTensor->getDimensionType());


    int total_correct = 0;
    int total_samples =0;
    double total_time =0;
    double perimage_time = 0;
    double total_pixels =0;
    double total_images =0;

    int num_tests = 5000/batch_size; // number of batches to run
    int imagesize = height*width; 
    for (int epoch = 0; epoch < num_tests; epoch++)
    {
    for (int b = 0; b < batch_size; ++b) {
        int img_id = epoch * batch_size + b;
        for (int j = 0; j < imagesize; ++j) {
            float pixel = images[(img_id* imagesize) + j] / 255.0f;
            pixel = (pixel - 0.1307f) / 0.3081f;
            inputHost.host<float>()[b * imagesize + j] = pixel;
            total_pixels++;
        }
    }

        // Copy data to device (MNN handles NCHW→NC4HW4)
        inputTensor->copyFromHostTensor(&inputHost);

        auto start = std::chrono::high_resolution_clock::now();
        net->runSession(session);
        auto end = std::chrono::high_resolution_clock::now();
        double inference_time_per_batch =
            std::chrono::duration<double, std::milli>(end - start).count();

    // Get output
        outputTensor->copyToHostTensor(&outputHost);

    // Now interpret output as [32 x 10]
    int batch_correct =0;
    double batch_time = 0.00;

    for (int b = 0; b < batch_size; ++b) {
        float max_val = -1e9;
        int predicted = -1;

        for (int k = 0; k < 10; ++k) {
            float val = outputHost.host<float>()[b * 10 + k];
            if (val > max_val) {
                max_val = val;
                predicted = k;
            }
        }

        int true_label = labels[epoch * batch_size + b];
        /*cout << "Sample " << b
            << " Predicted=" << predicted
            << " True=" << true_label
            << " maxval=" << max_val
            << " interference time= "<< inference_time_per_batch<<"ms"
            << "epoch number=" << epoch
            << endl;*/

        if (predicted == true_label) batch_correct++;
        batch_time += inference_time_per_batch;
    }
    auto batchaccuracy = (batch_correct * 100.0f / batch_size);
    //cout << "Batch: " <<epoch<< " accuracy =" << batchaccuracy << "% " <<" Avg Interference time per image: "<<(batch_time/batch_size)<<"ms"<< endl;
    total_correct += batch_correct;
    //total_samples += batch_size;
    total_time += inference_time_per_batch;
    //perimage_time += (batch_time/batch_size);

    }

    total_images = total_pixels/imagesize;
   /* cout<<"--------------------------------"<<endl;
    cout <<"Model Accuracy for" << " batchsize = "<< batch <<" is "<<((total_correct*100.0f)/total_samples)<<"%"<<endl;
    cout <<" Total interference time for the system is "<<total_time<<endl;
    cout<<" Avg per image intereference time "<<(perimage_time/num_tests)<<" for "<<total_samples<<" images"<<endl;*/  
    




    // ------------------------
    // Results
    // ------------------------
        std::cout << "\n==== BENCHMARK RESULTS ====\n";
        std::cout << "Batch size              : " << batch_size << "\n";
        std::cout << "Images processed        : " << total_images << "\n";
        std::cout << "Avg batch inference time: "
                  << (total_time / num_tests) << " ms\n";
        std::cout << "Per-image latency       : "
                  << total_time / total_images << " ms\n";
        std::cout << "Accuracy                : "
                  << (100.0 * total_correct / total_images) << " %\n";

    return 0;

    }


