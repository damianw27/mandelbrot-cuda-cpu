#define NO_FREETYPE
#define CONFIG_COUNT_2D 9

#include <cmath>
#include <chrono>
#include <iostream>
#include <pngwriter.h>
#include <cstdlib>
#include <fstream>

#ifdef NO_FREETYPE
using namespace std;
#endif

static const double COLORS[41][3] = {
        {1.0,  1.0,   1.0},
        {1.0,  1.0,   1.0},
        {1.0,  1.0,   1.0},
        {1.0,  1.0,   1.0},
        {1.0,  1.0,   1.0},
        {1.0,  0.7,   1.0},
        {1.0,  0.7,   1.0},
        {1.0,  0.7,   1.0},
        {0.97, 0.5,   0.94},
        {0.97, 0.5,   0.94},
        {0.94, 0.25,  0.88},
        {0.94, 0.25,  0.88},
        {0.91, 0.12,  0.81},
        {0.88, 0.06,  0.75},
        {0.85, 0.03,  0.69},
        {0.82, 0.015, 0.63},
        {0.78, 0.008, 0.56},
        {0.75, 0.004, 0.50},
        {0.72, 0.0,   0.44},
        {0.69, 0.0,   0.37},
        {0.66, 0.0,   0.31},
        {0.63, 0.0,   0.25},
        {0.60, 0.0,   0.19},
        {0.56, 0.0,   0.13},
        {0.53, 0.0,   0.06},
        {0.5,  0.0,   0.0},
        {0.47, 0.06,  0.0},
        {0.44, 0.12,  0},
        {0.41, 0.18,  0.0},
        {0.38, 0.25,  0.0},
        {0.35, 0.31,  0.0},
        {0.31, 0.38,  0.0},
        {0.28, 0.44,  0.0},
        {0.25, 0.50,  0.0},
        {0.22, 0.56,  0.0},
        {0.19, 0.63,  0.0},
        {0.16, 0.69,  0.0},
        {0.13, 0.75,  0.0},
        {0.06, 0.88,  0.0},
        {0.03, 0.94,  0.0},
        {0.0,  0.0,   0.0}
};

static const int CONFIG_2D_HORIZONTAL[CONFIG_COUNT_2D] = {2, 4, 8, 16, 32, 64, 128, 256, 512};
static const int CONFIG_2D_VERTICAL[CONFIG_COUNT_2D] = {2, 4, 8, 16, 32, 64, 128, 256, 512};
static const int CONFIG_1D[] = {32, 64, 128, 256, 512, 1024};

__global__ void cudaMandelbrot1(double x0, double y0, double x1, double y1, int width, int height, int iterationsCount, int *data) {
    double dX = (x1 - x0) / double(width - 1);
    double dY = (y1 - y0) / double(height - 1);
    int i;
    double x, y, Zx, Zy, tZx, tZy;
    int idX = int(threadIdx.x + blockIdx.x * blockDim.x);
    double tmpWidth, tmpHeight;
    int size = height * width;

    if (idX < size) {
        tmpWidth = (double) idX / (double) size;
        tmpHeight = double(idX % size);
        x = x0 + dX * tmpHeight;
        y = y0 + dY * tmpWidth;
        Zx = x;
        Zy = y;
        i = 0;

        while (i < iterationsCount && ((Zx * Zx + Zy * Zy) < 4)) {
            tZx = Zx * Zx - Zy * Zy + x;
            tZy = 2 * Zx * Zy + y;
            Zx = tZx;
            Zy = tZy;
            i++;
        }

        data[idX] = i;
    }
}

__global__ void cudaMandelbrot2(double x0, double y0, double x1, double y1, int width, int height, int iterationsCount, int *data) {
    double dX = (x1 - x0) / double(width - 1);
    double dY = (y1 - y0) / double(height - 1);
    int i;
    double x, y, Zx, Zy, tZx, tZy;
    auto tmpWidth = double((blockIdx.x * blockDim.x) + threadIdx.x);
    auto tmpHeight = double((blockIdx.y * blockDim.y) + threadIdx.y);

    if ((tmpWidth < (double) width) && (tmpHeight < (double) height)) {
        x = x0 + dX * tmpWidth;
        y = y0 + dY * tmpHeight;
        Zx = x;
        Zy = y;
        i = 0;

        while (i < iterationsCount && ((Zx * Zx + Zy * Zy) < 4)) {
            tZx = Zx * Zx - Zy * Zy + x;
            tZy = 2 * Zx * Zy + y;
            Zx = tZx;
            Zy = tZy;
            i++;
        }

        int index = int(tmpHeight * (double) width + tmpWidth);
        data[index] = i;
    }
}

int cpuMandelbrot1(double x0, double y0, double x1, double y1, int width, int height, int iterationsCount, int *data) {
    double dX = (x1 - x0) / double(width - 1);
    double dY = (y1 - y0) / double(height - 1);
    double x, y, Zx, Zy, tZx;
    int sum = 0;
    int i;

    for (int tmpWidth = 0; tmpWidth < height; tmpWidth++) {
        for (int tmpHeight = 0; tmpHeight < width; tmpHeight++) {
            x = x0 + (double) tmpHeight * dX;
            y = y0 + (double) tmpWidth * dY;
            Zx = x;
            Zy = y;
            i = 0;

            while ((i < iterationsCount) && ((Zx * Zx + Zy * Zy) < 4)) {
                tZx = Zx * Zx - Zy * Zy + x;
                Zy = 2 * Zx * Zy + y;
                Zx = tZx;

                i++;
            }

            int index = tmpWidth * width + tmpHeight;
            data[index] = i;
            sum += i;
        }
    }

    return sum;
}


int cpuMandelbrot2(double x0, double y0, double x1, double y1, int width, int height, int iterationsCount, int *data) {
    double dX = (x1 - x0) / double(width - 1);
    double dY = (y1 - y0) / double(height - 1);
    double x, y, Zx, Zy, tZx;
    int sum = 0;
    int i;
    int size = width * height;
    int tmpWidth, tmpHeight;

    for (int index = 0; index < size; index++) {
        tmpWidth = index / width;
        tmpHeight = index % width;
        x = x0 + (double) tmpHeight * dX;
        y = y0 + (double) tmpWidth * dY;
        Zx = x;
        Zy = y;
        i = 0;

        while ((i < iterationsCount) && ((Zx * Zx + Zy * Zy) < 4)) {
            tZx = Zx * Zx - Zy * Zy + x;
            Zy = 2 * Zx * Zy + y;
            Zx = tZx;

            i++;
        }

        data[index] = i;
        sum += i;
    }

    return sum;
}

void generatePicture(const int *data, int width, int height, int iterationsCount) {
    double red_value, green_value, blue_value;
    double scale = 256.0f / (double) iterationsCount;

    pngwriter png(width, height, 1.0, "mandelbrot_output.png");

    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            int colorIndex = (int) floor(5.0 * scale * log2(1.0 * (double) data[j * width + i] + 1));
            red_value = COLORS[colorIndex][0];
            green_value = COLORS[colorIndex][2];
            blue_value = COLORS[colorIndex][1];
            png.plot(i, j, red_value, green_value, blue_value);
        }
    }

    png.close();
}

int compare(const int *data1, const int *data2, int length) {
    int result = 0;
    int data1Overflow, data2Overflow;

    for (int index = 0; index < length; index++) {
        data1Overflow = (data1[index] > 255) ? 1 : 0;
        data2Overflow = (data2[index] > 255) ? 1 : 0;
        result += (int) data1Overflow == data2Overflow;
    }

    return result;
}

double mean(const double *data, int length) {
    double result = 0.0;

    for (int index = 0; index < length; index++) {
        result += data[index];
    }

    result = result / length;
    return result;
}

double min(const double *data, int length) {
    double minValue = data[0];

    for (int index = 1; index < length; index++) {
        if (data[index] >= minValue) {
            continue;
        }

        minValue = data[index];
    }

    return minValue;
}

int max(const int *data, int length) {
    int maxValue = data[0];

    for (int index = 1; index < length; index++) {
        if (data[index] <= maxValue) {
            continue;
        }

        maxValue = data[index];
    }

    return maxValue;
}

double median(const double *data, int length) {
    auto *location = new int[length];
    auto *sorted = new double[length];
    int currentValueScore;
    double currentValue;

    for (int outerIndex = 0; outerIndex < length; outerIndex++) {
        currentValueScore = 0;
        currentValue = data[outerIndex];

        for (int innerIndex = 0; innerIndex < length; innerIndex++) {
            if (data[innerIndex] < currentValue)
                currentValueScore++;
        }

        location[outerIndex] = currentValueScore;
    }

    for (int index = 0; index < length; index++) {
        sorted[location[index]] = data[index];
    }

    return sorted[length / 2];
}

double standardDeviation(double *data, int length) {
    double dataMeanValue = mean(data, length);
    double result = 0.0;

    for (int index = 0; index < length; index++) {
        result += (data[index] - dataMeanValue) * (data[index] - dataMeanValue);
    }

    result = sqrt(result / length / (length - 1));
    return result;
}

double acceleration(double minCudaTime, double cpuTime) {
    return cpuTime / minCudaTime;
}

void report2D(double *results, int localIterationsCount, int width, int height, double cpu_time) {
    double outMin;
    ofstream output("mandelbrot_report.csv", ofstream::out);
    output << "Threads; Median; Mean; SD; Min; Acceleration;" << endl;

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (CONFIG_2D_HORIZONTAL[i] * CONFIG_2D_VERTICAL[j] > 16 &&
                CONFIG_2D_HORIZONTAL[i] * CONFIG_2D_VERTICAL[j] < 1025) {
                outMin = min(&results[(j + i * width) * localIterationsCount], localIterationsCount);
                output << "(" << CONFIG_2D_HORIZONTAL[i] << " x " << CONFIG_2D_VERTICAL[j] << "); "
                       << median(&results[(j + i * width) * localIterationsCount], localIterationsCount) << "; "
                       << mean(&results[(j + i * width) * localIterationsCount], localIterationsCount) << "; "
                       << standardDeviation(&results[(j + i * width) * localIterationsCount], localIterationsCount) << "; "
                       << outMin << "; "
                       << acceleration(outMin, 1000 * cpu_time) << ";"
                       << endl;
            }
        }
    }

    output.close();
}

void report1D(double *results, int localIterationsCount, int configurationsCount, double cpuTime) {
    double outMin;
    ofstream output("mandelbrot_report.csv", ofstream::out);
    output << "Threads; Median; Mean; SD; Min; Acceleration;" << endl;

    for (int i = 0; i < configurationsCount; i++) {
        outMin = min(&results[i * localIterationsCount], localIterationsCount);
        output << "(" << CONFIG_1D[i] << "); "
               << median(&results[i * localIterationsCount], localIterationsCount) << "; "
               << mean(&results[i * localIterationsCount], localIterationsCount) << "; "
               << standardDeviation(&results[i * localIterationsCount], localIterationsCount) << "; "
               << outMin << "; "
               << acceleration(outMin, 1000 * cpuTime) << ";"
               << endl;
    }

    output.close();
}


int main(int argc, char **argv) {
    if (argc != 11) {
        cout << "------- MANDELBROT IMPLEMENTATION CUDA -------" << endl;
        cout << "ARG[0] - xStart" << endl;
        cout << "ARG[1] - yStart" << endl;
        cout << "ARG[2] - xEnd" << endl;
        cout << "ARG[3] - yEnd" << endl;
        cout << "ARG[4] - width" << endl;
        cout << "ARG[5] - height" << endl;
        cout << "ARG[6] - iterationsCount" << endl;
        cout << "ARG[7] - shouldCompareWithCPU" << endl;
        cout << "ARG[8] - shouldGenerateImage" << endl;
        cout << "ARG[9] - shouldUse2D" << endl;
        cout << "Example usage: ./mandelbrot_gpu -1. -1. 1. 1. 3000 3000 256 0 1 1 > output.txt" << endl;
        exit(1);
    }

    double x0 = stof(argv[1]);
    double y0 = stof(argv[2]);
    double x1 = stof(argv[3]);
    double y1 = stof(argv[4]);
    int width = stoi(argv[5]);
    int height = stoi(argv[6]);
    int iterationsCount = stoi(argv[7]);
    int shouldCompare = stoi(argv[8]);
    int shouldGenerateImage = stoi(argv[9]);
    int shouldUse2D = stoi(argv[10]);

    cudaError_t status;

    int outImageSize = width * height;
    int *mandel_data_host;
    int *mandel_data_device;
    int *mandel_data_cpu = new int[outImageSize];

    status = cudaMalloc((void **) &mandel_data_device, outImageSize * sizeof(int));

    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
    }

    status = cudaMallocHost((void **) &mandel_data_host, outImageSize * sizeof(int));

    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
    }

    cout << "Starting Mandelbrot (CUDA)" << endl;
    cout << "Corners - start = (" << x0 << ", " << y0 << "); end = (" << x1 << ", " << y1 << ");" << endl;

    auto startTime = chrono::steady_clock::now();

    int localIterationsCount;
    int configurationsCount;
    double *results;

    if (shouldUse2D) {
        cout << "Using version 2D of mandelbrot algorithm." << endl;

        localIterationsCount = 10;
        results = new double[CONFIG_COUNT_2D * CONFIG_COUNT_2D * localIterationsCount];
        int base, location;

        for (int c1 = 0; c1 < CONFIG_COUNT_2D; c1++) {
            base = CONFIG_COUNT_2D * c1 * localIterationsCount;

            for (int c2 = 0; c2 < CONFIG_COUNT_2D; c2++) {
                location = base + c2 * localIterationsCount;

                if (CONFIG_2D_HORIZONTAL[c1] * CONFIG_2D_VERTICAL[c2] > 16 &&
                    CONFIG_2D_HORIZONTAL[c1] * CONFIG_2D_VERTICAL[c2] < 1025) {
                    int blockWidth = CONFIG_2D_HORIZONTAL[c1];
                    int blockHeight = CONFIG_2D_VERTICAL[c2];
                    dim3 threadsPerBlock(blockWidth, blockHeight, 1);
                    dim3 numBlocks(width / blockWidth + 1, height / blockHeight + 1, 1);

                    for (int index = 0; index < localIterationsCount; index++) {
                        auto executionStart = chrono::steady_clock::now();
                        cudaMandelbrot2<<<numBlocks, threadsPerBlock, 0>>>(x0, y0, x1, y1, width, height,
                                                                           iterationsCount, mandel_data_device);
                        status = cudaDeviceSynchronize();

                        if (status != cudaSuccess) {
                            cout << cudaGetErrorString(status) << endl;
                        }

                        auto executionTimePoint = chrono::steady_clock::now() - executionStart;
                        double executionTime = chrono::duration<double, milli>(executionTimePoint).count();

                        cout << "Thread = " << "(" << CONFIG_2D_HORIZONTAL[c1] << ", " << CONFIG_2D_VERTICAL[c2] << ")"
                             << "; Iteration = " << index
                             << "; Time = " << executionTime
                             << "; Base = " << base
                             << "; Location = " << location
                             << endl;

                        results[location + index] = executionTime;
                    }
                } else {
                    for (int i = 0; i < localIterationsCount; i++) {
                        results[location + i] = 1.0 * (i + 1);
                    }
                }
            }
        }
    } else {
        cout << "Using version 1D of mandelbrot algorithm." << endl;

        localIterationsCount = 15;
        configurationsCount = 0;
        results = new double[localIterationsCount * 6];

        for (int currentThreadsCount = 32; currentThreadsCount < 2048; currentThreadsCount = 2 * currentThreadsCount) {
            dim3 threadsPerBlock(currentThreadsCount, 1, 1);
            dim3 numBlocks(width * height / currentThreadsCount + 1, 1, 1);

            for (int index = 0; index < localIterationsCount; index++) {
                auto executionStart = chrono::steady_clock::now();
                cudaMandelbrot1<<<numBlocks, threadsPerBlock, 0>>>(x0, y0, x1, y1, width, height, iterationsCount,
                                                                   mandel_data_device);
                status = cudaDeviceSynchronize();

                if (status != cudaSuccess) {
                    cout << cudaGetErrorString(status) << endl;
                }

                auto executionStop = chrono::steady_clock::now();
                auto executionTimePoint = executionStop - executionStart;
                auto executionTime = chrono::duration<double, milli>(executionTimePoint).count();

                cout << "Thread " << currentThreadsCount
                     << "; Iteration " << index
                     << "; Time " << executionTime << " ms"
                     << endl;

                results[configurationsCount * localIterationsCount + index] = executionTime;
            }

            configurationsCount++;
        }
    }

    status = cudaMemcpy(mandel_data_host, mandel_data_device, outImageSize * sizeof(int), cudaMemcpyDeviceToHost);

    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
    }

    auto endTime = chrono::steady_clock::now();
    auto fullComputationTimePoint = endTime - startTime;
    auto fullComputationTime = chrono::duration<double, milli>(fullComputationTimePoint).count() / 1000;
    cout << "Computation and data transfer ended in " << fullComputationTime << "s" << endl;

    if (shouldGenerateImage == 1) {
        int maxValue = max(mandel_data_host, outImageSize);
        auto pictureGenerationStart = chrono::steady_clock::now();
        generatePicture(mandel_data_host, width, height, maxValue);
        auto pictureGenerationEnd = chrono::steady_clock::now();
        auto pictureGenerationTimePoint = pictureGenerationEnd - pictureGenerationStart;
        auto pictureGenerationTime = chrono::duration<double, milli>(pictureGenerationTimePoint).count() / 1000;
        cout << "Generation of image ended in " << pictureGenerationTime << "s" << endl;
    }

    status = cudaFree(mandel_data_device);

    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
    }

    if (shouldCompare == 1) {
        auto cpuTimeStart = chrono::steady_clock::now();

        if (shouldUse2D == 1) {
            cpuMandelbrot2(x0, y0, x1, y1, width, height, iterationsCount, mandel_data_cpu);
        } else {
            cpuMandelbrot1(x0, y0, x1, y1, width, height, iterationsCount, mandel_data_cpu);
        }

        auto cpuTimeEnd = chrono::steady_clock::now();
        auto cpuTimePoint = cpuTimeEnd - cpuTimeStart;
        auto cpuTime = 1.0 * chrono::duration<double, milli>(cpuTimePoint).count() / 1000;
        int pixelsCount = height * width;
        int samePixelsCount = compare(mandel_data_host, mandel_data_cpu, pixelsCount);

        cout << "Comparing CUDA with CPU ended in " << cpuTime << "s" << endl;
        cout << "Comparison result in pixels: " << samePixelsCount << " out of " << pixelsCount << "pixels." << endl;
        cout << "Comparison result in percentage: " << 100.0 * samePixelsCount / height / width << "%" << endl;

        if (shouldUse2D == 1) {
            report2D(results, localIterationsCount, CONFIG_COUNT_2D, CONFIG_COUNT_2D, cpuTime);
        } else {
            report1D(results, localIterationsCount, configurationsCount, cpuTime);
        }
    }

    status = cudaFreeHost(mandel_data_host);

    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
    }
}
