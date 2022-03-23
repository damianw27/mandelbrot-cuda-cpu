# Mandelbrot CUDA/CPU Implementation

## Arguments
- xStart
- yStart
- xEnd
- yEnd
- width
- height
- iterationsCount
- shouldCompareWithCPU (not included in CPU version)
- shouldGenerateImage
- shouldUse2D

## Execution command:

### For GPU version
> ./mandelbrot_gpu -1. -1. 1. 1. 3000 3000 256 0 1 1 > output.txt

### For CPU version
> ./mandelbrot_cpu -1. -1. 1. 1. 3000 3000 256 1 1 > output.txt
