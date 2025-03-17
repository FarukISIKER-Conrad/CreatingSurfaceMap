#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <opencv2/opencv.hpp> // Only for imshow function

// Define color stops from the gradient in the image
const struct ColorStop {
    float position;  // 0.0 to 1.0
    unsigned char r, g, b;
} JET_COLORS[] = {
    {0.0f,   0xFD, 0x00, 0x15}, // Red at 0%
    {0.2f,   0xFE, 0xFC, 0x38}, // Yellow at 20%
    {0.5f,   0x24, 0xAF, 0x38}, // Green at 50%
    {0.73f,  0x1D, 0xFC, 0xFE}, // Cyan at 73%
    {1.0f,   0x00, 0x2E, 0xF8}  // Blue at 100%
};

const int NUM_COLOR_STOPS = 5;

// Structure to hold RGB values
struct Pixel {
    unsigned char r, g, b;
};

// BMP file header structure
#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t fileType;      // File type, must be 0x4D42 ('BM')
    uint32_t fileSize;      // Size of the file in bytes
    uint16_t reserved1;     // Reserved, must be 0
    uint16_t reserved2;     // Reserved, must be 0
    uint32_t dataOffset;    // Offset to image data in bytes
};

struct BMPInfoHeader {
    uint32_t headerSize;    // Size of this header in bytes (40)
    int32_t width;          // Width of the image
    int32_t height;         // Height of the image
    uint16_t planes;        // Number of color planes (must be 1)
    uint16_t bitsPerPixel;  // Number of bits per pixel (16 for RGB565)
    uint32_t compression;   // Compression method (0 for BI_RGB, 3 for BI_BITFIELDS)
    uint32_t imageSize;     // Size of the image data in bytes
    int32_t xPixelsPerMeter;// X pixels per meter (resolution)
    int32_t yPixelsPerMeter;// Y pixels per meter (resolution)
    uint32_t colorsUsed;    // Number of colors in the color palette
    uint32_t colorsImportant;// Number of important colors
};

struct BMPColorMasks {
    uint32_t redMask;       // Red channel mask (0xF800 for RGB565)
    uint32_t greenMask;     // Green channel mask (0x07E0 for RGB565)
    uint32_t blueMask;      // Blue channel mask (0x001F for RGB565)
    uint32_t alphaMask;     // Alpha channel mask (0x0000 for RGB565)
};
#pragma pack(pop)

// Create a custom resize function using bicubic interpolation
std::vector<std::vector<float>> resizeBicubic(const std::vector<std::vector<float>>& input, int newWidth, int newHeight) {
    int oldWidth = input[0].size();
    int oldHeight = input.size();
    
    std::vector<std::vector<float>> output(newHeight, std::vector<float>(newWidth, 0.0f));
    
    float scaleX = static_cast<float>(oldWidth) / newWidth;
    float scaleY = static_cast<float>(oldHeight) / newHeight;
    
    auto cubic = [](float x) -> float {
        float absx = std::abs(x);
        if (absx <= 1.0f) {
            return 1.5f * absx * absx * absx - 2.5f * absx * absx + 1.0f;
        } else if (absx < 2.0f) {
            return -0.5f * absx * absx * absx + 2.5f * absx * absx - 4.0f * absx + 2.0f;
        }
        return 0.0f;
    };
    
    for (int y = 0; y < newHeight; y++) {
        for (int x = 0; x < newWidth; x++) {
            float srcX = x * scaleX;
            float srcY = y * scaleY;
            
            int x0 = static_cast<int>(std::floor(srcX));
            int y0 = static_cast<int>(std::floor(srcY));
            
            float sum = 0.0f;
            float weightSum = 0.0f;
            
            for (int dy = -1; dy <= 2; dy++) {
                for (int dx = -1; dx <= 2; dx++) {
                    int nx = x0 + dx;
                    int ny = y0 + dy;
                    
                    // Clamp to edge
                    nx = std::max(0, std::min(oldWidth - 1, nx));
                    ny = std::max(0, std::min(oldHeight - 1, ny));
                    
                    float weight = cubic(srcX - nx) * cubic(srcY - ny);
                    sum += input[ny][nx] * weight;
                    weightSum += weight;
                }
            }
            
            output[y][x] = sum / weightSum;
        }
    }
    
    return output;
}

// Function to apply custom colormap based on the image's gradient
Pixel applyCustomColormap(float value) {
    // Ensure value is between 0 and 1
    value = std::max(0.0f, std::min(1.0f, value));
    
    // Find the appropriate color stops
    int idx1 = 0;
    int idx2 = 1;
    
    for (int i = 0; i < NUM_COLOR_STOPS - 1; i++) {
        if (value >= JET_COLORS[i].position && value <= JET_COLORS[i+1].position) {
            idx1 = i;
            idx2 = i + 1;
            break;
        }
    }
    
    // Calculate interpolation factor between the two colors
    float range = JET_COLORS[idx2].position - JET_COLORS[idx1].position;
    float factor = range > 0 ? (value - JET_COLORS[idx1].position) / range : 0;
    
    // Interpolate RGB values
    Pixel result;
    result.r = static_cast<unsigned char>(JET_COLORS[idx1].r * (1.0f - factor) + JET_COLORS[idx2].r * factor);
    result.g = static_cast<unsigned char>(JET_COLORS[idx1].g * (1.0f - factor) + JET_COLORS[idx2].g * factor);
    result.b = static_cast<unsigned char>(JET_COLORS[idx1].b * (1.0f - factor) + JET_COLORS[idx2].b * factor);
    
    return result;
}

// Function to convert RGB888 to RGB565
uint16_t convertToRGB565(unsigned char r, unsigned char g, unsigned char b) {
    // Convert 8-bit to 5-bit for red (mask out the lower 3 bits)
    uint16_t r5 = (r >> 3) & 0x1F;
    
    // Convert 8-bit to 6-bit for green (mask out the lower 2 bits)
    uint16_t g6 = (g >> 2) & 0x3F;
    
    // Convert 8-bit to 5-bit for blue (mask out the lower 3 bits)
    uint16_t b5 = (b >> 3) & 0x1F;
    
    // Combine into a 16-bit value (RGB565 format)
    // Format: RRRR RGGG GGGB BBBB
    return (r5 << 11) | (g6 << 5) | b5;
}

// Function to save RGB565 data as a BMP file
void saveRGB565AsBMP(const std::vector<std::vector<uint16_t>>& rgb565Data, const std::string& filename) {
    int width = rgb565Data[0].size();
    int height = rgb565Data.size();
    
    // BMP rows are padded to 4-byte boundaries
    int rowPadding = (width * 2) % 4 == 0 ? 0 : 4 - ((width * 2) % 4);
    int dataSize = (width * 2 + rowPadding) * height;
    
    // Create file headers
    BMPFileHeader fileHeader = {
        0x4D42,                              // 'BM' signature
        static_cast<uint32_t>(54 + 12 + dataSize), // File size (54 bytes header + 12 bytes color masks + data)
        0,                                   // Reserved
        0,                                   // Reserved
        54 + 12                              // Offset to pixel data (54 bytes header + 12 bytes color masks)
    };
    
    BMPInfoHeader infoHeader = {
        40,                                  // Header size
        width,                               // Width
        -height,                             // Height (negative for top-down)
        1,                                   // Planes
        16,                                  // Bits per pixel
        3,                                   // Compression (BI_BITFIELDS)
        static_cast<uint32_t>(dataSize),     // Image size
        2835,                                // X pixels per meter (72 DPI)
        2835,                                // Y pixels per meter (72 DPI)
        0,                                   // Colors used
        0                                    // Important colors
    };
    
    BMPColorMasks colorMasks = {
        0xF800,                              // Red mask (5 bits)
        0x07E0,                              // Green mask (6 bits)
        0x001F,                              // Blue mask (5 bits)
        0x0000                               // Alpha mask (0 bits)
    };
    
    // Open file for writing
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening BMP file for writing" << std::endl;
        return;
    }
    
    // Write headers
    file.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));
    file.write(reinterpret_cast<const char*>(&colorMasks), sizeof(colorMasks));
    
    // Write pixel data
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint16_t pixel = rgb565Data[y][x];
            file.write(reinterpret_cast<const char*>(&pixel), 2);
        }
        
        // Write row padding
        if (rowPadding > 0) {
            const char padding[4] = {0, 0, 0, 0};
            file.write(padding, rowPadding);
        }
    }
    
    file.close();
    std::cout << "RGB565 data saved as BMP file: " << filename << std::endl;
}

int main() {
    // Initialize the data matrix manually (4x5)
    std::vector<std::vector<float>> data = {
        {4095.0f, 4095.0f, 0.0f, 2048.0f, 2048.0f},
        {4095.0f, 2048.0f, 2048.0f, 2048.0f, 4095.0f},
        {4095.0f, 4095.0f, 4095.0f, 4095.0f, 4095.0f},
        {4095.0f, 4095.0f, 4095.0f, 4095.0f, 4095.0f}
    };
    
    // Normalize the matrix manually
    std::vector<std::vector<float>> scaledData(data.size(), std::vector<float>(data[0].size()));
    for (size_t i = 0; i < data.size(); i++) {
        for (size_t j = 0; j < data[i].size(); j++) {
            scaledData[i][j] = data[i][j] / 4095.0f;
        }
    }
    
    // Print the scaled data
    std::cout << "Scaled data:" << std::endl;
    for (const auto& row : scaledData) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    
    // Resize the data using custom bicubic interpolation function
    std::vector<std::vector<float>> resizedHeatmap = resizeBicubic(scaledData, 226, 226);
    
    // Convert to 8-bit and apply colormap
    std::vector<std::vector<Pixel>> heatmap(226, std::vector<Pixel>(226));
    for (int i = 0; i < 226; i++) {
        for (int j = 0; j < 226; j++) {
            heatmap[i][j] = applyCustomColormap(resizedHeatmap[i][j]);
        }
    }
    
    // Save final matrix values to a text file
    std::ofstream file("heatmap_values.txt");
    if (file.is_open()) {
        for (size_t i = 0; i < resizedHeatmap.size(); i++) {
            for (size_t j = 0; j < resizedHeatmap[i].size(); j++) {
                file << resizedHeatmap[i][j] << " "; // Space-separated
            }
            file << std::endl; // New row
        }
        file.close();
        std::cout << "Heatmap values saved to heatmap_values.txt" << std::endl;
    } else {
        std::cerr << "Error opening file!" << std::endl;
    }
    
    // Create an OpenCV Mat from our heatmap data for display only
    cv::Mat displayImage(226, 226, CV_8UC3);
    for (int i = 0; i < 226; i++) {
        for (int j = 0; j < 226; j++) {
            displayImage.at<cv::Vec3b>(i, j)[0] = heatmap[i][j].b;
            displayImage.at<cv::Vec3b>(i, j)[1] = heatmap[i][j].g;
            displayImage.at<cv::Vec3b>(i, j)[2] = heatmap[i][j].r;
        }
    }
    
    // Convert to RGB565 format
    std::vector<std::vector<uint16_t>> rgb565Data(226, std::vector<uint16_t>(226));
    for (int i = 0; i < 226; i++) {
        for (int j = 0; j < 226; j++) {
            rgb565Data[i][j] = convertToRGB565(
                heatmap[i][j].r,
                heatmap[i][j].g,
                heatmap[i][j].b
            );
        }
    }
    
    // Save RGB565 data as BMP file
    saveRGB565AsBMP(rgb565Data, "heatmap_rgb565.bmp");
    
    // Additionally save RGB565 data as hex values in a text file for reference
    std::ofstream hexFile("heatmap_rgb565_hex.txt");
    if (hexFile.is_open()) {
        hexFile << std::hex;
        for (size_t i = 0; i < rgb565Data.size(); i++) {
            for (size_t j = 0; j < rgb565Data[i].size(); j++) {
                hexFile << "0x" << std::setfill('0') << std::setw(4) 
                       << rgb565Data[i][j] << " ";
            }
            hexFile << std::endl;
        }
        hexFile.close();
        std::cout << "RGB565 hex values saved to heatmap_rgb565_hex.txt" << std::endl;
    } else {
        std::cerr << "Error opening hex file!" << std::endl;
    }
    
    // Show the result (using OpenCV as requested)
    cv::imshow("Heatmap", displayImage);
    cv::waitKey(0);
    
    return 0;
}