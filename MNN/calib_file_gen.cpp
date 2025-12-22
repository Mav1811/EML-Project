#include <fstream>
#include <vector>
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <limits.h>
#include <filesystem>

int32_t readInt(std::ifstream& f)
{
    unsigned char b[4];
    f.read((char*)b, 4);
    return (b[0]<<24)|(b[1]<<16)|(b[2]<<8)|b[3];
}

int main()
{
    // Create calib directory first
    std::filesystem::path calibDir("calib_RGB");
    if (!std::filesystem::exists(calibDir))
    {
        bool created = std::filesystem::create_directory(calibDir);
        std::cout << "create_directory returned: " << created << std::endl;
    }

    std::ifstream img("/home/orangepi/Documents/data/MNIST/raw/train-images-idx3-ubyte", std::ios::binary);
    if (!img.is_open())
    {
        std::cerr << "Failed to open MNIST image file\n";
        return -1;
    }

    int magic = readInt(img);
    int numImages = readInt(img);
    int rows = readInt(img);
    int cols = readInt(img);

    std::cout << "Images: " << numImages << "  Size: "
              << rows << "x" << cols << std::endl;

    for (int n = 0; n < 200; n++)  // 200 images for calibration
    {
        img.seekg(16 + n * rows * cols);

        std::vector<unsigned char> raw(rows * cols);
        img.read((char*)raw.data(), raw.size());

        // Create RGB tensor (3 channels)
        std::vector<float> tensor(3 * rows * cols);

        for (int i = 0; i < rows * cols; i++)
        {
            float v = raw[i] / 255.0f;
            v = (v - 0.1307f) / 0.3081f;
            // replicate grayscale into R,G,B
            tensor[i * 3 + 0] = v; // R
            tensor[i * 3 + 1] = v; // G
            tensor[i * 3 + 2] = v; // B
        }

        std::ofstream out(calibDir / (std::to_string(n) + ".jpg"), std::ios::binary);
        out.write((char*)tensor.data(), tensor.size() * sizeof(float));
    }

    std::cout << "Calibration data generated successfully\n";

    // Optional: read first file to check
    std::ifstream file("calib/0.bin", std::ios::binary);
    std::vector<float> data(3 * rows * cols);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));

    std::cout << "First 10 RGB values (flattened):\n";
    for (int i = 0; i < 30; i++) // 10 pixels × 3 channels
        std::cout << data[i] << " ";
    std::cout << std::endl;

    return 0;
}
