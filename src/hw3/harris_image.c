#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#include "matrix.h"
#include <time.h>

// Frees an array of descriptors.
// descriptor *d: the array.
// int n: number of elements in array.
void free_descriptors(descriptor *d, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(d[i].data);
    }
    free(d);
}

// Create a feature descriptor for an index in an image.
// image im: source image.
// int i: index in image for the pixel we want to describe.
// returns: descriptor for that index.
descriptor describe_index(image im, int i)
{
    int w = 5;
    descriptor d;
    d.p.x = i%im.w;
    d.p.y = i/im.w;
    d.data = calloc(w*w*im.c, sizeof(float));
    d.n = w*w*im.c;
    int c, dx, dy;
    int count = 0;
    // If you want you can experiment with other descriptors
    // This subtracts the central value from neighbors
    // to compensate some for exposure/lighting changes.
    for(c = 0; c < im.c; ++c){
        float cval = im.data[c*im.w*im.h + i];
        for(dx = -w/2; dx < (w+1)/2; ++dx){
            for(dy = -w/2; dy < (w+1)/2; ++dy){
                float val = get_pixel(im, i%im.w+dx, i/im.w+dy, c);
                d.data[count++] = cval - val;
            }
        }
    }
    return d;
}

// Marks the spot of a point in an image.
// image im: image to mark.
// ponit p: spot to mark in the image.
void mark_spot(image im, point p)
{
    int x = p.x;
    int y = p.y;
    int i;
    for(i = -9; i < 10; ++i){
        set_pixel(im, x+i, y, 0, 1);
        set_pixel(im, x, y+i, 0, 1);
        set_pixel(im, x+i, y, 1, 0);
        set_pixel(im, x, y+i, 1, 0);
        set_pixel(im, x+i, y, 2, 1);
        set_pixel(im, x, y+i, 2, 1);
    }
}

// Marks corners denoted by an array of descriptors.
// image im: image to mark.
// descriptor *d: corners in the image.
// int n: number of descriptors to mark.
void mark_corners(image im, descriptor *d, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        mark_spot(im, d[i].p);
    }
}

// Creates a 1d Gaussian filter.
// float sigma: standard deviation of Gaussian.
// returns: single row image of the filter.
image make_1d_gaussian(float sigma)
{
    const int center = ((int) 6 * sigma + 1) / 2;
    const int kernel_size = 2 * center + 1;
    image kernel = make_image(kernel_size, 1, 1);
    
    // Fill the values
    for (int u = 0; u < kernel_size; ++u) {
        const double value = 1 / sqrt(TWOPI * sigma) * exp(-1 * (pow((double) u - center, 2) / (2 * sigma * sigma)));
        set_pixel(kernel, u, 0, 0, (float) value);
    }

    l1_normalize(kernel);
    return kernel;
}

// Smooths an image using separable Gaussian filter.
// image im: image to smooth.
// float sigma: std dev. for Gaussian.
// returns: smoothed image.
image smooth_image(image im, float sigma)
{
    image gaussian_kernel = make_1d_gaussian(sigma);
    

    // Create the horizontal kernel
    image gaussian_kernel_v = make_image(1, gaussian_kernel.w, 1);
    for (int i = 0; i < gaussian_kernel.w; ++i) {
        gaussian_kernel_v.data[i] = gaussian_kernel.data[i];
    }

    // Do 2 convolutions
    image conv_1 = convolve_image(im, gaussian_kernel, 1);
    image conv_2 = convolve_image(conv_1, gaussian_kernel_v, 1);

    return conv_2;
}

// Calculate the structure matrix of an image.
// image im: the input image.
// float sigma: std dev. to use for weighted sum.
// returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
//          third channel is IxIy.
image structure_matrix(image im, float sigma)
{
    image S = make_image(im.w, im.h, 3);
    // Gradient images
    const image gx_filter = make_gx_filter();
    const image gx = convolve_image(im, gx_filter, 0);

    const image gy_filter = make_gy_filter();
    const image gy = convolve_image(im, gy_filter, 0);

    for (int u = 0; u < im.w; ++u) {
        for (int v = 0; v < im.h; ++v) {
            const float Ix = get_pixel(gx, u, v, 0);
            const float Iy = get_pixel(gy, u, v, 0);
            set_pixel(S, u, v, 0, Ix*Ix);
            set_pixel(S, u, v, 1, Iy*Iy);
            set_pixel(S, u, v, 2, Ix*Iy);
        }
    }

    image smoothed_image = smooth_image(S, sigma);
    return smoothed_image;
}

// Estimate the cornerness of each pixel given a structure matrix S.
// image S: structure matrix for an image.
// returns: a response map of cornerness calculations.
image cornerness_response(image S)
{
    image R = make_image(S.w, S.h, 1);
    // We'll use formulation det(S) - alpha * trace(S)^2, alpha = .06.
    for (int u = 0; u < S.w; ++u) {
        for (int v = 0; v < S.h; ++v) {
            const float a = get_pixel(S, u, v, 0);
            const float d = get_pixel(S, u, v, 1);
            const float b = get_pixel(S, u, v, 2);

            const float det = a*d - b*b;
            const float trace = a+d;
            const float alpha = 0.06;
            const float cornerness = det - alpha * trace * trace;

            set_pixel(R, u, v, 0, cornerness);
        }
    }
    return R;
}

// Perform non-max supression on an image of feature responses.
// image im: 1-channel image of feature responses.
// int w: distance to look for larger responses.
// returns: image with only local-maxima responses within w pixels.
image nms_image(image im, int w)
{
    image r = copy_image(im);
    // for every pixel in the image:
    for (int u = 0; u < im.w; ++u) {
        for (int v = 0; v < im.h; ++v) {
            const float response = get_pixel(im, u, v, 0);

            // for neighbors within w
            for (int nms_u = u - w; nms_u <= u + w; ++nms_u) {
                for (int nms_v = v - w; nms_v <= v + w; ++nms_v) {
                    const float neighbor_response = get_pixel(im, nms_u, nms_v, 0);
                    if (neighbor_response > response) {
                        set_pixel(r, u, v, 0, -9999);
                        break;
                        break;
                    }
                }
            }
        }
    }

    return r;
}

// Perform harris corner detection and extract features from the corners.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
// int *n: pointer to number of corners detected, should fill in.
// returns: array of descriptors of the corners in the image.
descriptor *harris_corner_detector(image im, float sigma, float thresh, int nms, int *n)
{
    // Calculate structure matrix
    image S = structure_matrix(im, sigma);

    // Estimate cornerness
    image R = cornerness_response(S);

    // Run NMS on the responses
    image Rnms = nms_image(R, nms);

    // Count responses over a threshold
    int count = 0;
    for (int i = 0; i < Rnms.w * Rnms.h * Rnms.c; ++i) {
        if (Rnms.data[i] > thresh) {
            count += 1;
        }
    }

    
    *n = count; // <- set *n equal to number of corners in image.
    descriptor *d = calloc(count, sizeof(descriptor));
    
    int descriptor_idx = 0;
    for (int i = 0; i < Rnms.w * Rnms.h * Rnms.c; ++i) {
        if (Rnms.data[i] > thresh) {
            d[descriptor_idx] = describe_index(im, i);
            descriptor_idx += 1;
        }
    }


    free_image(S);
    free_image(R);
    free_image(Rnms);
    return d;
}

// Find and draw corners on an image.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
void detect_and_draw_corners(image im, float sigma, float thresh, int nms)
{
    int n = 0;
    descriptor *d = harris_corner_detector(im, sigma, thresh, nms, &n);
    mark_corners(im, d, n);
}
