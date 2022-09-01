typedef struct SurfHF {
    int2 p0, p1, p2, p3;
    float w;
} SurfHF;

float calcHaarPattern(read_only image2d_t inputImage, const int2 p, constant SurfHF f[], int N) {
    float d = 0;
    for (int k = 0; k < N; k++) {
        constant SurfHF* fk = &f[k];
        d += (read_imagef(inputImage, p + fk->p0).x +
              read_imagef(inputImage, p + fk->p3).x -
              read_imagef(inputImage, p + fk->p1).x -
              read_imagef(inputImage, p + fk->p2).x) * fk->w;
    }
    return d;
}

kernel void calcDetAndTrace(read_only image2d_t sumImage,
                            write_only image2d_t detImage,
                            write_only image2d_t traceImage,
                            int sampleStep,
                            constant SurfHF Dx[3],
                            constant SurfHF Dy[3],
                            constant SurfHF Dxy[4]) {
    const int2 imageCoordinates = (int2) (get_global_id(0), get_global_id(1));
    const int2 p = imageCoordinates * sampleStep;

    const float dx = calcHaarPattern(sumImage, p, Dx, 3);
    const float dy = calcHaarPattern(sumImage, p, Dy, 3);
    const float dxy = calcHaarPattern(sumImage, p, Dxy, 4);

    write_imagef(detImage, imageCoordinates, dx * dy - 0.81f * dxy * dxy);
    write_imagef(traceImage, imageCoordinates, dx + dy);
}
