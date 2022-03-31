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

const double colors[41][3] = {
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

static int config_2d_width[CONFIG_COUNT_2D] = {2, 4, 8, 16, 32, 64, 128, 256, 512};
static int config_2d_height[CONFIG_COUNT_2D] = {2, 4, 8, 16, 32, 64, 128, 256, 512};

__global__ void
cudaMandelbrot(double x0, double y0, double x1, double y1, int width, int height, int iterationsCount, int *data) {
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

__global__ void
cudaMandelbrot2(double x0, double y0, double x1, double y1, int width, int height, int iterationsCount, int *data) {
    double dX = (x1 - x0) / double(width - 1);
    double dY = (y1 - y0) / double(height - 1);
    int i;
    double x, y, Zx, Zy, tZx, tZy;
    double tmpWidth = double((blockIdx.x * blockDim.x) + threadIdx.x);
    double tmpHeight = double((blockIdx.y * blockDim.y) + threadIdx.y);

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

int
computeMandelbrot(double x0, double y0, double x1, double y1, int width, int height, int iterationsCount, int *data) {
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


int
computeMandelbrot2(double x0, double y0, double x1, double y1, int width, int height, int iterationsCount, int *data) {
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

void makePicturePNG(const int *data, int width, int height, int iterationsCount) {
    double red_value, green_value, blue_value;
    double scale = 256.0f / (double) iterationsCount;

    pngwriter png(width, height, 1.0, "mandelbrot_output.png");

    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            int colorIndex = (int) floor(5.0 * scale * log2f(1.0f * (double) data[j * width + i] + 1));
            red_value = colors[colorIndex][0];
            green_value = colors[colorIndex][2];
            blue_value = colors[colorIndex][1];
            png.plot(i, j, red_value, green_value, blue_value);
        }
    }

    png.close();
}

int compare(const int *data1, const int *data2, int length) {
    int sum = 0;
    int in1, in2;

    for (int i = 0; i < length; i++) {
        in1 = (data1[i] > 255) ? 1 : 0;
        in2 = (data2[i] > 255) ? 1 : 0;
        sum += (int) in1 == in2;
    }

    return sum;
}

double mean2(double *vec, int len) {
    double res = 0.0;

    for (int i = 0; i < len; i++) {
        res += vec[i];
    }

    res = res / len;
    return res;
}

double min2(double *vec, int len) {
    double min = vec[0];

    for (int i = 1; i < len; i++) {
        if (vec[i] >= min) {
            continue;
        }

        min = vec[i];
    }

    return min;
}

double median2(double *vec, int len) {
    int *location;
    double *sorted;
    int score;
    double curr;

    location = (int *) malloc(len * sizeof(int));
    sorted = (double *) malloc(len * sizeof(double));

    for (int i = 0; i < len; i++) {
        score = 0;
        curr = vec[i];
        for (int j = 0; j < len; j++) {
            if (vec[j] < curr) score++;
        }
        location[i] = score;
    }
    for (int i = 0; i < len; i++) {
        sorted[location[i]] = vec[i];
    }

    return sorted[len / 2];
}

double standardDeviation2(double *vec, int len) {
    double avg = mean2(vec, len);
    double res = 0.0;

    for (int i = 0; i < len; i++) {
        res += (vec[i] - avg) * (vec[i] - avg);
    }

    res = sqrt(res / len / (len - 1));
    return res;
}

double acceleration2(double Par, double Scal) {
    return Scal / Par;
}

void report2D(double *X, int reps, int wid, int hei, double CPU_time) {
    double outMin;
    ofstream output("mandelbrot_report.csv", ofstream::out);
    output << "Threads; Median; Mean; SD; Min; Acceleration;" << endl;

    for (int i = 0; i < wid; i++) {
        for (int j = 0; j < hei; j++) {
            if (config_2d_width[i] * config_2d_height[j] > 16 && config_2d_width[i] * config_2d_height[j] < 1025) {
                outMin = min2(&X[(j + i * wid) * reps], reps);
                output << "(" << config_2d_width[i] << " x " << config_2d_height[j] << "); "
                       << median2(&X[(j + i * wid) * reps], reps) << "; "
                       << mean2(&X[(j + i * wid) * reps], reps) << "; "
                       << standardDeviation2(&X[(j + i * wid) * reps], reps) << "; "
                       << outMin << "; "
                       << acceleration2(outMin, 1000 * CPU_time) << ";"
                       << endl;
            }
        }
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

    int *mandel_data_host;
    int *mandel_data_device;
    int *mandel_data_cpu = (int *) malloc(sizeof(int) * width * height);

    status = cudaMalloc((void **) &mandel_data_device, width * height * sizeof(int));

    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
    }

    status = cudaMallocHost((void **) &mandel_data_host, width * height * sizeof(int));

    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
    }

    time_t start, end;

    cout << "Starting Mandelbrot (CUDA)" << endl;
    cout << "Corners - start = (" << x0 << ", " << y0 << "); end = (" << x1 << ", " << y1 << ");" << endl;

    start = clock();

    if (shouldUse2D) {
        cout << "Using version 2D of mandelbrot algorithm." << endl;
//        cudaMandelbrot2<<<numBlocks, threadsPerBlock, 0>>>(x0, y0, x1, y1, width, height, iterationsCount,mandel_data_device);

        int reps = 10;
        double *results2D = new double[CONFIG_COUNT_2D * CONFIG_COUNT_2D * reps];
        int base, loc;

        for (int c1 = 0; c1 < CONFIG_COUNT_2D; c1++) {
            base = CONFIG_COUNT_2D * c1 * reps;

            for (int c2 = 0; c2 < CONFIG_COUNT_2D; c2++) {
                loc = base + c2 * reps;

                if (config_2d_width[c1] * config_2d_height[c2] > 16 &&
                    config_2d_width[c1] * config_2d_height[c2] < 1025) {
                    int blockWidth = config_2d_width[c1];
                    int blockHeight = config_2d_height[c2];
                    dim3 threadsPerBlock(blockWidth, blockHeight, 1);
                    dim3 numBlocks(width / blockWidth + 1, height / blockHeight + 1, 1);

                    for (int i = 0; i < reps; i++) {
                        start = clock();
                        auto singleBlockStartTime = chrono::steady_clock::now();
                        cudaMandelbrot2<<<numBlocks, threadsPerBlock, 0>>>(x0, y0, x1, y1, width, height,
                                                                           iterationsCount, mandel_data_device);
                        status = cudaDeviceSynchronize();

                        if (status != cudaSuccess) {
                            cout << cudaGetErrorString(status) << endl;
                        };

                        auto singleBlockExecutionTime = chrono::steady_clock::now() - singleBlockStartTime;
                        results2D[loc + i] = chrono::duration<double, milli>(singleBlockExecutionTime).count();
                    }
                } else {
                    for (int i = 0; i < reps; i++) {
                        results2D[loc + i] = 1.0 * (i + 1);
                    }
                }
            }
        }

        report2D(results2D, reps, CONFIG_COUNT_2D, CONFIG_COUNT_2D, 0);

    } else {
//        cudaMandelbrot<<<numBlocks, threadsPerBlock, 0>>>(x0, y0, x1, y1, width, height, iterationsCount,
//                                                          mandel_data_device);
    }

    status = cudaDeviceSynchronize();

    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
    }

    auto stop = chrono::steady_clock::now();

    status = cudaMemcpy(mandel_data_host, mandel_data_device, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
    }

    end = clock();

    cout << "Computation and data transfer ended in " << (double) (end - start) / CLOCKS_PER_SEC << "s" << endl;

    if (shouldGenerateImage == 1) {
        start = clock();
        makePicturePNG(mandel_data_host, width, height, iterationsCount);
        end = clock();
        cout << "Generation of image ended in " << (double) (end - start) / CLOCKS_PER_SEC << "s" << endl;
    }

    status = cudaFree(mandel_data_device);
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
    }

    if (shouldCompare == 1) {
        start = clock();

        if (shouldUse2D == 1) {
            computeMandelbrot2(x0, y0, x1, y1, width, height, iterationsCount, mandel_data_cpu);
        } else {
            computeMandelbrot(x0, y0, x1, y1, width, height, iterationsCount, mandel_data_cpu);
        }

        end = clock();

        int pixelsCount = height * width;
        int samePixelsCount = compare(mandel_data_host, mandel_data_cpu, pixelsCount);
        cout << "Comparing CUDA with CPU ended in " << (double) (end - start) / CLOCKS_PER_SEC << "s" << endl;
        cout << "Comparison result in pixels: " << samePixelsCount << " out of " << pixelsCount << "pixels." << endl;
        cout << "Comparison result in percentage: " << 100.0 * samePixelsCount / height / width << "%" << endl;
    }

    status = cudaFreeHost(mandel_data_host);

    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
    }
}
