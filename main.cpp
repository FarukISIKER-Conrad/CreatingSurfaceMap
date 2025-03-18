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

void showImage(const std::vector<std::vector<float>>& data, const std::string& windowName = "Image") {
    if (data.empty() || data[0].empty()) {
        std::cerr << "Error: Empty data!" << std::endl;
        return;
    }

    int rows = data.size();
    int cols = data[0].size();

    // Convert vector to cv::Mat
    cv::Mat img(rows, cols, CV_32F);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            img.at<float>(i, j) = data[i][j];

    // Normalize to 0-255 range
    cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
    img.convertTo(img, CV_8U);

    // Show image
    cv::imshow(windowName, img);
    cv::waitKey(0);
}
std::vector<std::vector<float>> matToVector2D(const cv::Mat& mat) {
    std::vector<std::vector<float>> vec(mat.rows, std::vector<float>(mat.cols));

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            vec[i][j] = mat.at<float>(i, j);
        }
    }
    return vec;
}
cv::Mat vector2DToMat(const std::vector<std::vector<float>>& vec) {
    if (vec.empty()) return cv::Mat(); // Return an empty Mat if input is empty

    int rows = vec.size();
    int cols = vec[0].size();
    cv::Mat mat(rows, cols, CV_32F); // Create a float matrix

    for (int i = 0; i < rows; i++) {
        memcpy(mat.ptr<float>(i), vec[i].data(), cols * sizeof(float)); // Efficient row copy
    }

    return mat;
}
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

// Add this function before applyCustomColormap and call it in the color mapping process
float remapValue(float value) {
    // Apply a non-linear transformation to give more weight to low values
    // This will expand the red area and compress the blue area
    if (value < 0.5f) {
        // Expand the lower half (red to green)
        return value * 0.7f;
    } else {
        // Compress the upper half (green to blue)
        return 0.35f + (value - 0.5f) * 1.3f;
    }
}
// Create a custom resize function using bicubic interpolation

// Interpolation kernel
float u(float s, float a) {
    if ((std::abs(s) >= 0) && (std::abs(s) <= 1)) {
        return (a + 2) * std::pow(std::abs(s), 3) - (a + 3) * std::pow(std::abs(s), 2) + 1;
    } else if ((std::abs(s) > 1) && (std::abs(s) <= 2)) {
        return a * std::pow(std::abs(s), 3) - (5 * a) * std::pow(std::abs(s), 2) + (8 * a) * std::abs(s) - 4 * a;
    }
    return 0;
}

// Padded image creation
std::vector<std::vector<float>> padding(const std::vector<std::vector<float>>& img) {
    int H = img.size();
    int W = img[0].size();
    
    // Create padded image with zeros
    std::vector<std::vector<float>> zimg(H + 4, std::vector<float>(W + 4, 0.0f));
    
    // Copy original image to center
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            zimg[h + 2][w + 2] = img[h][w];
        }
    }
    
    // Pad the first/last two columns
    for (int h = 0; h < H; h++) {
        // Left columns
        zimg[h + 2][0] = img[h][0];
        zimg[h + 2][1] = img[h][0];
        
        // Right columns
        zimg[h + 2][W + 2] = img[h][W - 1];
        zimg[h + 2][W + 3] = img[h][W - 1];
    }
    
    // Pad the first/last two rows
    for (int w = 0; w < W; w++) {
        // Top rows
        zimg[0][w + 2] = img[0][w];
        zimg[1][w + 2] = img[0][w];
        
        // Bottom rows
        zimg[H + 2][w + 2] = img[H - 1][w];
        zimg[H + 3][w + 2] = img[H - 1][w];
    }
    
    // Pad the corners
    // Top-left
    zimg[0][0] = img[0][0];
    zimg[0][1] = img[0][0];
    zimg[1][0] = img[0][0];
    zimg[1][1] = img[0][0];
    
    // Top-right
    zimg[0][W + 2] = img[0][W - 1];
    zimg[0][W + 3] = img[0][W - 1];
    zimg[1][W + 2] = img[0][W - 1];
    zimg[1][W + 3] = img[0][W - 1];
    
    // Bottom-left
    zimg[H + 2][0] = img[H - 1][0];
    zimg[H + 2][1] = img[H - 1][0];
    zimg[H + 3][0] = img[H - 1][0];
    zimg[H + 3][1] = img[H - 1][0];
    
    // Bottom-right
    zimg[H + 2][W + 2] = img[H - 1][W - 1];
    zimg[H + 2][W + 3] = img[H - 1][W - 1];
    zimg[H + 3][W + 2] = img[H - 1][W - 1];
    zimg[H + 3][W + 3] = img[H - 1][W - 1];
    
    return zimg;
}

// Bicubic matrix multiplication
float computeBicubic(const std::vector<float>& row_kernel, 
                     const std::vector<std::vector<float>>& pixel_values, 
                     const std::vector<float>& col_kernel) {
    // First multiply row_kernel with pixel_values
    std::vector<float> temp(4, 0.0f);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            temp[i] += row_kernel[j] * pixel_values[j][i];
        }
    }
    
    // Then multiply with col_kernel
    float result = 0.0f;
    for (int i = 0; i < 4; i++) {
        result += temp[i] * col_kernel[i];
    }
    
    return result;
}

// Bicubic interpolation function with requested signature
std::vector<std::vector<float>> resizeBicubic(const std::vector<std::vector<float>>& input, int newWidth, int newHeight) {
    // Get original dimensions
    int H = input.size();
    int W = input[0].size();
    
    // Calculate scale ratios
    float ratioH = static_cast<float>(newHeight) / H;
    float ratioW = static_cast<float>(newWidth) / W;
    
    // Padding the image
    std::vector<std::vector<float>> paddedInput = padding(input);
    
    // Create output image
    std::vector<std::vector<float>> output(newHeight, std::vector<float>(newWidth, 0.0f));
    
    // Bicubic parameter
    float a = -0.5f;
    
    // Inverse scaling factors
    float h_h = 1.0f / ratioH;
    float h_w = 1.0f / ratioW;
    
    std::cout << "Starting bicubic interpolation..." << std::endl;
    
    // Process each pixel in the output image
    for (int j = 0; j < newHeight; j++) {
        for (int i = 0; i < newWidth; i++) {
            // Get the corresponding position in the original image
            float y = j * h_h + 1.5f; // +2 because of padding
            float x = i * h_w + 1.5f;
            
            // Calculate kernel weights for x
            float x1 = 1 + x - std::floor(x);
            float x2 = x - std::floor(x);
            float x3 = std::floor(x) + 1 - x;
            float x4 = std::floor(x) + 2 - x;
            
            // Calculate kernel weights for y
            float y1 = 1 + y - std::floor(y);
            float y2 = y - std::floor(y);
            float y3 = std::floor(y) + 1 - y;
            float y4 = std::floor(y) + 2 - y;
            
            // Kernel vectors
            std::vector<float> row_kernel = {u(x1, a), u(x2, a), u(x3, a), u(x4, a)};
            std::vector<float> col_kernel = {u(y1, a), u(y2, a), u(y3, a), u(y4, a)};
            
            // 4x4 grid of pixel values around the interpolation point
            std::vector<std::vector<float>> pixel_values = {
                {
                    paddedInput[static_cast<int>(y-y1)][static_cast<int>(x-x1)],
                    paddedInput[static_cast<int>(y-y2)][static_cast<int>(x-x1)],
                    paddedInput[static_cast<int>(y+y3)][static_cast<int>(x-x1)],
                    paddedInput[static_cast<int>(y+y4)][static_cast<int>(x-x1)]
                },
                {
                    paddedInput[static_cast<int>(y-y1)][static_cast<int>(x-x2)],
                    paddedInput[static_cast<int>(y-y2)][static_cast<int>(x-x2)],
                    paddedInput[static_cast<int>(y+y3)][static_cast<int>(x-x2)],
                    paddedInput[static_cast<int>(y+y4)][static_cast<int>(x-x2)]
                },
                {
                    paddedInput[static_cast<int>(y-y1)][static_cast<int>(x+x3)],
                    paddedInput[static_cast<int>(y-y2)][static_cast<int>(x+x3)],
                    paddedInput[static_cast<int>(y+y3)][static_cast<int>(x+x3)],
                    paddedInput[static_cast<int>(y+y4)][static_cast<int>(x+x3)]
                },
                {
                    paddedInput[static_cast<int>(y-y1)][static_cast<int>(x+x4)],
                    paddedInput[static_cast<int>(y-y2)][static_cast<int>(x+x4)],
                    paddedInput[static_cast<int>(y+y3)][static_cast<int>(x+x4)],
                    paddedInput[static_cast<int>(y+y4)][static_cast<int>(x+x4)]
                }
            };
            
            // Compute interpolated value using bicubic algorithm
            output[j][i] = computeBicubic(row_kernel, pixel_values, col_kernel);
        }
    }
    
    std::cout << "Bicubic interpolation complete." << std::endl;
    
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


int main() {
    // // Initialize the data matrix manually (4x5)
    // std::vector<std::vector<float>> data = {
    //     {0.0f   , 0.0f  , 0.0f   ,  0.0f   },
    //     {2048.0f, 2048.0f, 2048.0f, 2048.0f},
    //     {0.0f   , 0.0f  , 0.0f   ,  0.0f   }
    // };
    
    // Initialize the data matrix with a wider range of values
    std::vector<std::vector<float>> data = {
        {0.0f   , 0.0f  , 0.0f      ,0.0f },  
        {0.0f   , 0.0f  , 0.0f      ,0.0f },     // Row 1: All zeros (normalized to 0.0) - RED
        {0.0f   , 2048.0f  , 0.0f   ,0.0f},     // Row 2: All 2048 (normalized to 0.5) - GREEN
        {0.0f   , 0.0f  , 0.0f      ,0.0f } 
    };
    // Normalize the matrix manually
    std::vector<std::vector<float>> normalizedData(data.size(), std::vector<float>(data[0].size()));
    for (size_t i = 0; i < data.size(); i++) {
        for (size_t j = 0; j < data[i].size(); j++) {
            normalizedData[i][j] = data[i][j] / 4095.0f;
        }
    }
    //showImage(scaledData,"scaledData");
    // Print the scaled data
    std::cout << "Scaled data:" << std::endl;
    for (const auto& row : normalizedData) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    

    cv::Mat normalizedDataMat;
    cv::Mat resizedHeatmapMat;
    // Resize the data using custom bicubic interpolation function
    std::vector<std::vector<float>> resizedHeatmap = resizeBicubic(normalizedData, 226, 226);
    showImage(resizedHeatmap,"resizedHeatmap Without CV");

    normalizedDataMat = vector2DToMat(normalizedData);
    cv::resize(normalizedDataMat, resizedHeatmapMat, cv::Size(226, 226), 0, 0, cv::INTER_CUBIC);
    
    resizedHeatmap = matToVector2D(resizedHeatmapMat);
    showImage(resizedHeatmap,"resizedHeatmap with cv");





    // Convert to 8-bit and apply colormap
    std::vector<std::vector<Pixel>> heatmap(226, std::vector<Pixel>(226));
    for (int i = 0; i < 226; i++) {
        for (int j = 0; j < 226; j++) {
            // Then in your code, modify the applyCustomColormap call:
            heatmap[i][j] = applyCustomColormap((resizedHeatmap[i][j]));
        }
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
   
    
    // Show the result (using OpenCV as requested)
    cv::imshow("Heatmap", displayImage);
    cv::waitKey(0);
    
    return 0;
}