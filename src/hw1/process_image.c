#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

float get_pixel(image im, int x, int y, int c)
{
    // Bounds checking
    if (x < 0) {
        x = 0;
    } else if (x >= im.w) {
        x = im.w - 1;
    }

    if (y < 0) {
        y = 0;
    } else if (y >= im.h) {
        y = im.h - 1;
    }

    if (c < 0) {
        c = 0;
    } else if (c >= im.c) {
        c = im.c - 1;
    }

    // for (int i = 0; i < im.h * im.w * im.c; ++i) {
        // printf("i: %i Value: %f\n", i, im.data[i]);
    // }

    // printf("-----");
    const size_t idx = (c * im.w * im.h) + (y * im.w) + x;
    // printf("Using actual index x %i y %i c %i idx %zu \n", x, y, c, idx);

    // printf("Return %f \n", im.data[idx]);
    return im.data[idx];
}

void set_pixel(image im, int x, int y, int c, float v)
{
    // Bounds checking
    if (x < 0 || x >= im.w || y < 0 || y >= im.h || c < 0 || c >= im.c) {
        return;
    }
    const size_t idx = (c * im.w * im.h) + (y * im.w) + x;
    im.data[idx] = v;
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    const size_t image_size = im.w * im.h * im.c * sizeof(float);
    memcpy(copy.data, im.data, image_size);
    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);
    for (int u = 0; u < im.w; ++u) {
        for (int v = 0; v < im.h; ++v) {
            const float gray_value = 0.299 * get_pixel(im, u, v, 0) + 
                                     0.587 * get_pixel(im, u, v, 1) + 
                                     0.114 * get_pixel(im, u, v, 2);
            set_pixel(gray, u, v, 0, gray_value);
        }
    }
    return gray;
}

void shift_image(image im, int c, float v)
{
    for (int i = 0; i < im.w; ++i) {
        for (int j = 0; j < im.h; ++j) {
            const float curr_value = get_pixel(im, i, j, c);
            set_pixel(im, i, j, c, curr_value + v);
        }
    }
}

void scale_image(image im, int c, float v) {
    for (int i = 0; i < im.w; ++i) {
        for (int j = 0; j < im.h; ++j) {
            const float curr_value = get_pixel(im, i, j, c);
            set_pixel(im, i, j, c, curr_value * v);
        }
    }
}

void clamp_image(image im)
{
    for (int i = 0; i < im.w * im.h * im.c; ++i) {
        if (im.data[i] > 1) {
            im.data[i] = 1;
        } else if (im.data[i] < 0) {
            im.data[i] = 0;
        }
    }
}


// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
    int counter = 0;
    for (int u = 0; u < im.w; ++u) {
        for (int v = 0; v < im.h; ++v) {
            const float curr_r = get_pixel(im, u, v, 0);
            const float curr_g = get_pixel(im, u, v, 1);
            const float curr_b = get_pixel(im, u, v, 2);

            // Calculate value
            const float value = fmax(curr_b, fmax(curr_g, curr_r));
            const float minimum = fmin(curr_b, fmin(curr_g, curr_r));
            const float C = value - minimum;
            
            // Calculate saturation
            float saturation = 0.0;
            if (value > 0) {
                saturation = C / value;
            }

            // Calculate hue
            float hue = 0.0;
            if (C == 0) {
                hue = 0.0;
            } else if (value == curr_r) {
                hue = (curr_g - curr_b) / C;

            } else if (value == curr_g) {
                hue = (curr_b - curr_r) / C + 2;

            } else if (value == curr_b) {
                hue = (curr_r - curr_g) / C + 4;
            }

            if (hue < 0) {
                hue = hue / 6 + 1;
            } else if (hue > 0) {
                hue = hue / 6;
            }

            set_pixel(im, u, v, 0, hue);
            set_pixel(im, u, v, 1, saturation);
            set_pixel(im, u, v, 2, value);

            counter++;
        }
    }
}

void hsv_to_rgb(image im)
{
    for (int u = 0; u < im.w; ++u) {
        for (int v = 0; v < im.h; ++v) {
            float hue = get_pixel(im, u, v, 0);
            const float saturation = get_pixel(im, u, v, 1);
            const float V = get_pixel(im, u, v, 2);

            hue = hue * 6;
            const int Hi = (int) hue;
            const float F = hue - Hi;

            const float P = V * (1 - saturation);
            const float Q = V * (1 - F * saturation);
            const float T = V * (1 - (1 - F) * saturation);

            float curr_r, curr_g, curr_b;

            if (Hi == 0) {
                curr_r = V;
                curr_g = T;
                curr_b = P;
            
            } else if (Hi == 1) {
                curr_r = Q;
                curr_g = V;
                curr_b = P;

            } else if (Hi == 2) {
                curr_r = P;
                curr_g = V;
                curr_b = T;

            } else if (Hi == 3) {
                curr_r = P;
                curr_g = Q;
                curr_b = V;

            } else if (Hi == 4) {
                curr_r = T;
                curr_g = P;
                curr_b = V;

            } else {
                curr_r = V;
                curr_g = P;
                curr_b = Q;
            }

            set_pixel(im, u, v, 0, curr_r);
            set_pixel(im, u, v, 1, curr_g);
            set_pixel(im, u, v, 2, curr_b);
        }
    }
}
