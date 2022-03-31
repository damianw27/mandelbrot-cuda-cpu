#define NO_FREETYPE

#include <cmath>
#include <chrono>
#include <iostream>
#include <pngwriter.h>
#include <cstdlib>

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

int computeMandelbrot(double x0, double y0, double x1, double y1, int width, int height, int iterationsCount, int *data) {
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


int computeMandelbrot2(double x0, double y0, double x1, double y1, int width, int height, int iterationsCount, int *data) {
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

int main(int argc, char **argv) {
    if (argc != 10) {
        cout << "-- MANDELBROT IMPLEMENTATION CPU --" << endl;
        cout << "ARG[0] - xStart" << endl;
        cout << "ARG[1] - yStart" << endl;
        cout << "ARG[2] - xEnd" << endl;
        cout << "ARG[3] - yEnd" << endl;
        cout << "ARG[4] - width" << endl;
        cout << "ARG[5] - height" << endl;
        cout << "ARG[6] - iterationsCount" << endl;
        cout << "ARG[7] - shouldGenerateImage" << endl;
        cout << "ARG[8] - shouldUse2D" << endl;
        cout << "Example usage: ./mandelbrot_cpu -1. -1. 1. 1. 3000 3000 256 1 1 > output.txt" << endl;
        exit(1);
    }

    double x0 = stof(argv[1]);
    double y0 = stof(argv[2]);
    double x1 = stof(argv[3]);
    double y1 = stof(argv[4]);
    int width = stoi(argv[5]);
    int height = stoi(argv[6]);
    int iterationsCount = stoi(argv[7]);
    int shouldGenerateImage = stoi(argv[8]);
    int shouldUse2D = stoi(argv[9]);
    int *mandel_data = (int *) malloc(sizeof(int) * width * height);

    cout << "Starting Mandelbrot (CPU)" << endl;
    cout << "Corners - start = (" << x0 << ", " << y0 << "); end = (" << x1 << ", " << y1 << ");" << endl;

    time_t start = clock();

    if (shouldUse2D) {
        cout << "Using version 2D of mandelbrot algorithm." << endl;
        computeMandelbrot(x0, y0, x1, y1, width, height, iterationsCount, mandel_data);
    } else {
        computeMandelbrot2(x0, y0, x1, y1, width, height, iterationsCount, mandel_data);
    }

    time_t end = clock();

    cout << "Computation ended in " << (double) (end - start) / CLOCKS_PER_SEC << "s" << endl;

    if (shouldGenerateImage == 1) {
        start = clock();
        makePicturePNG(mandel_data, width, height, iterationsCount);
        end = clock();

        cout << "Generation of image ended in " << (double) (end - start) / CLOCKS_PER_SEC << "s" << endl;
    }
}