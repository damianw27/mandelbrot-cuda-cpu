#define NO_FREETYPE

#include <cmath>
#include <chrono>
#include <iostream>
#include <pngwriter.h>
#include <cstdlib>

#ifdef NO_FREETYPE
using namespace std;
#endif

const float colors[41][3] = {
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

__global__ void cudaMandelbrot(float x0, float y0, float x1, float y1, int width, int height, int iterationsCount, int *data) {
    float dX = (x1 - x0) / float(width - 1);
    float dY = (y1 - y0) / float(height - 1);
    int i;
    float x, y, Zx, Zy, tZx, tZy;
    int idX = int(threadIdx.x + blockIdx.x * blockDim.x);
    float tmpWidth, tmpHeight;
    int size = height * width;

    if (idX < size) {
        tmpWidth = (float) idX / (float) size;
        tmpHeight = float(idX % size);
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

__global__ void cudaMandelbrot2(float x0, float y0, float x1, float y1, int width, int height, int iterationsCount, int *data) {
    float dX = (x1 - x0) / float(width - 1);
    float dY = (y1 - y0) / float(height - 1);
    int i;
    float x, y, Zx, Zy, tZx, tZy;
    float tmpWidth = float((blockIdx.x * blockDim.x) + threadIdx.x);
    float tmpHeight = float((blockIdx.y * blockDim.y) + threadIdx.y);

    if ((tmpWidth < (float) width) && (tmpHeight < (float) height)) {
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

        int index = int(tmpHeight * (float) width + tmpWidth);
        data[index] = i;
    }
}

void makePicturePNG(const int *data, int width, int height, int iterationsCount) {
    float red_value, green_value, blue_value;
    float scale = 256.0f / (float) iterationsCount;

    pngwriter png(width, height, 1.0, "mandelbrot_output.png");

    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            int colorIndex = (int) floor(5.0 * scale * log2f(1.0f * (float) data[j * width + i] + 1));
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

int main(int argc, char **argv) {
    if (argc != 11) {
        printf("Wywołanie %s LD_Re, LD_Im, PG_Re, PG_Im, Poziom, Pion, Iteracje, Compare  Picture \n", argv[0]);
        printf("Flagi:  Compare: 0/1 - porównaj rezultat z CPU lub nie\n");
        printf("Flagi:  Picture: 0/1 - generuj obrazki lub nie \n");
        exit(1);
    }

    float x0 = stof(argv[1]);
    float y0 = stof(argv[2]);
    float x1 = stof(argv[3]);
    float y1 = stof(argv[4]);
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

    printf("Corners - (%lf , %lf) and ", x0, y0);
    printf("(%lf , %lf)\n", x1, y1);

    int block_width = 8;
    int block_height = 32;
    dim3 threadsPerBlock(block_width, block_height, 1);
    dim3 numBlocks(width / block_width + 1, height / block_height + 1, 1);

    start = clock();
    auto start2 = chrono::steady_clock::now();

    if (shouldUse2D) {
        cudaMandelbrot2<<<numBlocks, threadsPerBlock, 0>>>(x0, y0, x1, y1, width, height, iterationsCount,mandel_data_device);
    } else {
        cudaMandelbrot<<<numBlocks, threadsPerBlock, 0>>>(x0, y0, x1, y1, width, height, iterationsCount,mandel_data_device);
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

    auto diff = stop - start2;

    cout << "Kernel " << chrono::duration<float, milli>(diff).count() << " ms" << endl;
    cout << "Kernel " << chrono::duration<float, micro>(diff).count() << " us" << endl;
    cout << "Kernel " << chrono::duration<float, nano>(diff).count() << " ns" << endl;

    printf("Start %f End %f clock ticks\n", (float) start, (float) end);
    printf("Computations and transfer %lf s\n\n", 1.0f * (float) (end - start) / CLOCKS_PER_SEC);

    if (shouldGenerateImage == 1) {
        start = clock();
        makePicturePNG(mandel_data_host, width, height, iterationsCount);
        end = clock();
        printf("Picture production took %lf s\n\n", 1.0f * (float) (end - start) / CLOCKS_PER_SEC);
    }

    status = cudaFree(mandel_data_device);
    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
    }

    if (shouldCompare == 1) {
        printf("Computing reference\n");
        start = clock();
//        int SUM = computeMandelbrot(x0, y0, x1, y1, width, height, iterationsCount, mandel_data_cpu);
        end = clock();
        printf("Time %lf s\n\n", 1.0f * (float) (end - start) / CLOCKS_PER_SEC);
        int ident = compare(mandel_data_host, mandel_data_cpu, height * width);
        printf("%d out of %d pixels are identical (%8.2lf) %% \n", ident, height * width, 100.0 * ident / height / width);
    }

    status = cudaFreeHost(mandel_data_host);

    if (status != cudaSuccess) {
        cout << cudaGetErrorString(status) << endl;
    }
}
