#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>

#include "image.h"
#define TWOPI 6.2831853

/******************************** Resizing *****************************
  To resize we'll need some interpolation methods and a function to create
  a new image and fill it in with our interpolation methods.
************************************************************************/

float nn_interpolate(image im, float x, float y, int c)
{
    /***********************************************************************
      This function performs nearest-neighbor interpolation on image "im"
      given a floating column value "x", row value "y" and integer channel "c",
      and returns the interpolated value.
    ************************************************************************/
    const int rounded_x = round(x);
    const int rounded_y = round(y);

    return get_pixel(im, rounded_x, rounded_y, c);
}

image nn_resize(image im, int w, int h)
{
    /***********************************************************************
      This function uses nearest-neighbor interpolation on image "im" to a new
      image of size "w x h"
    ************************************************************************/
    image new_image = make_image(w, h, im.c);

    for (int u = 0; u < w; u++) {
        for (int v = 0; v < h; v++) {
            for (int c = 0; c  < im.c; c++) {
                const float mapped_u = (u + 0.5) * im.w / (float) w - 0.5;
                const float mapped_v = (v + 0.5) * im.h / (float) h - 0.5;
                const float new_pixel_value = nn_interpolate(im, mapped_u, mapped_v, c);
                set_pixel(new_image, u, v, c, new_pixel_value);
            }
        }
    }

    return new_image;
}

float bilinear_interpolate(image im, float x, float y, int c)
{
    /***********************************************************************
      This function performs bilinear interpolation on image "im" given
      a floating column value "x", row value "y" and integer channel "c".
      It interpolates and returns the interpolated value.
    ************************************************************************/
    // Get the upper left pixel coordinate
    float x_base;
    if (x < 0) {
        x_base = (int) x - 1;
    } else {
        x_base = (int) x;
    }

    float y_base;
    if (y < 0) {
        y_base = (int) y - 1;
    } else {
        y_base = (int) y;
    }

    const float x_left = x - x_base;
    const float x_right = 1 - x_left;
    const float y_top = y - y_base;
    const float y_bottom = 1 - y_top;

    const float A4 = x_left * y_top;
    const float A3 = x_left * y_bottom;
    const float A2 = x_right * y_top;
    const float A1 = x_right * y_bottom;


    const float ret = A1 * get_pixel(im, x_base, y_base, c) + 
           A2 * get_pixel(im, x_base, y_base + 1, c) +
           A3 * get_pixel(im, x_base + 1, y_base, c) + 
           A4 * get_pixel(im, x_base + 1, y_base + 1, c);

    return ret;
}

image bilinear_resize(image im, int w, int h)
{
    /***********************************************************************
      This function uses bilinear interpolation on image "im" to a new image
      of size "w x h". Algorithm is same as nearest-neighbor interpolation.
    ************************************************************************/
    image new_image = make_image(w, h, im.c);

    for (int u = 0; u < w; u++) {
        for (int v = 0; v < h; v++) {
            for (int c = 0; c  < im.c; c++) {
                const float mapped_u = (u + 0.5) * im.w / (float) w - 0.5;
                const float mapped_v = (v + 0.5) * im.h / (float) h - 0.5;
                const float new_pixel_value = bilinear_interpolate(im, mapped_u, mapped_v, c);
                set_pixel(new_image, u, v, c, new_pixel_value);
            }
        }
    }

    return new_image;
}


/********************** Filtering: Box filter ***************************
  We want to create a box filter. We will only use square box filters.
************************************************************************/

void l1_normalize(image im)
{
    /***********************************************************************
      This function divides each value in image "im" by the sum of all the
      values in the image and modifies the image in place.
    ************************************************************************/
    float normalization_amount = 0;
    for (int u = 0; u < im.w; ++u) {
        for (int j = 0; j < im.h; ++j) {
            for (int c = 0; c < im.c; ++c) {
                normalization_amount += get_pixel(im, u, j, c);
            }
        }
    }

    for (int u = 0; u < im.w; ++u) {
        for (int j = 0; j < im.h; ++j) {
            for (int c = 0; c < im.c; ++c) {
                const float new_value = get_pixel(im, u, j, c) / normalization_amount;
                set_pixel(im, u, j, c, new_value);
            }
        }
    }
}

image make_box_filter(int w)
{
    /***********************************************************************
      This function makes a square filter of size "w x w". Make an image of
      width = height = w and number of channels = 1, with all entries equal
      to 1. Then use "l1_normalize" to normalize your filter.
    ************************************************************************/
    image im = make_image(w, w, 1);
    for (int u = 0; u < im.w; ++u) {
        for (int j = 0; j < im.h; ++j) {
            for (int c = 0; c < im.c; ++c) {
                set_pixel(im, u, j, c, 1);
            }
        }
    }
    l1_normalize(im);
    return im;
}

image convolve_image(image im, image filter, int preserve)
{
    /***********************************************************************
      This function convolves the image "im" with the "filter". The value
      of preserve is 1 if the number of input image channels need to be 
      preserved. Check the detailed algorithm given in the README.  
    ************************************************************************/
    assert((filter.c == 1 || filter.c == im.c));
    assert(filter.w == filter.h && filter.w % 2 == 1);  // Square and odd

    image new_image = make_image(im.w, im.h, im.c);

    for (int u = 0; u < im.w; ++u) {
        for (int v = 0; v < im.h; ++v) {
            for (int c = 0; c < im.c; ++c) {
                // Do the convolution sum
                int filter_c = (filter.c == 1) ? 0 : c;
                float conv_sum = 0.0;
                int filter_offset = (int) filter.w / 2;
                for (int x = 0; x < filter.w; ++x) {
                    for (int y = 0; y < filter.h; ++y) {
                        conv_sum += get_pixel(filter, x, y, filter_c) * get_pixel(im, u - filter_offset + x, v - filter_offset + y, c);
                    }
                }
                set_pixel(new_image, u, v, c, conv_sum);
            }
        }
    }

    // Sum over the channel axis if necessary
    if (!preserve) {
        image summed_image = make_image(new_image.w, new_image.h, 1);

        for (int u = 0; u < im.w; ++u) {
            for (int v = 0; v < im.h; ++v) {
                float summed_value = 0.0;
                for (int c = 0; c < im.c; ++c) {
                    summed_value += get_pixel(new_image, u, v, c);
                }

                set_pixel(summed_image, u, v, 0, summed_value);
            }
        }

        free_image(new_image);
        return summed_image;
    }

    return new_image;
}

image make_highpass_filter()
{
    /***********************************************************************
      Create a 3x3 filter with highpass filter values using image.data[]
    ************************************************************************/
    image highpass = make_image(3, 3, 1);
    const float highpass_data[] = {0.0, -1.0, 0.0, -1.0, 4.0,
                                   -1.0, 0.0, -1.0, 0.0};
    for (int i = 0; i < 9; ++i) {
        highpass.data[i] = highpass_data[i];
    }

    return highpass;
}

image make_sharpen_filter()
{
    /***********************************************************************
      Create a 3x3 filter with sharpen filter values using image.data[]
    ************************************************************************/
    image sharpen = make_image(3, 3, 1);
    const float sharpen_data[] = {0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0};
    for (int i = 0; i < 9; ++i) {
        sharpen.data[i] = sharpen_data[i];
    }

    return sharpen;
}

image make_emboss_filter()
{
    /***********************************************************************
      Create a 3x3 filter with emboss filter values using image.data[]
    ************************************************************************/
    image emboss = make_image(3, 3, 1);
    const float emboss_data[] = {-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0};
    for (int i = 0; i < 9; ++i) {
        emboss.data[i] = emboss_data[i];
    }

    return emboss;
}

// Question 2.3.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?

// We should preserve channels when the filter should be run independently on the different color channels.
// Preserved filters are sharpen, and emboss. Non-preserved one is highpass because we want the frequency
// across the 3 color channels combined.

// Question 2.3.2: Do we have to do any post-processing for the above filters? Which ones and why?

// We need to clamp all three of them between 0 and 1 to produce a valid image. The result can be outside
// the allowed range of the images.

image make_gaussian_filter(float sigma)
{
    // TODO
    /***********************************************************************
      sigma: a float number for the Gaussian.
      Create a Gaussian filter with the given sigma. Note that the kernel size 
      is the next highest odd integer from 6 x sigma. Return the Gaussian filter.
    ************************************************************************/
    const int center = ((int) 6 * sigma + 1) / 2;
    const double sigma_prec = (double) sigma;
    // printf("Center %i\n", center);
    const int kernel_size = 2 * center + 1;
    image kernel = make_image(kernel_size, kernel_size, 1);
    for (int u = 0; u < kernel_size; ++u) {
        for (int v = 0; v < kernel_size; ++v) { 
            const double value = 1 / (TWOPI * sigma_prec * sigma_prec) * exp(-1 * (pow((double) u - center, 2) + pow((double) v  - center, 2)) / (2 * sigma_prec * sigma_prec));
            set_pixel(kernel, u, v, 0, (float) value);
        }
    }

    l1_normalize(kernel);
    return kernel;
}

image add_image(image a, image b)
{
    // TODO
    /***********************************************************************
      The input images a and image b have the same height, width, and channels.
      Sum the given two images and return the result, which should also have
      the same height, width, and channels as the inputs. Do necessary checks.
    ************************************************************************/
    assert(a.w == b.w && a.h == b.h && a.c == b.c);
    image output_image = make_image(a.w, a.h, a.c);
    for (int u = 0; u < a.w; ++u) {
        for (int v = 0; v < a.h; ++v) {
            for (int c = 0; c < a.c; ++c) {
                const float sum = get_pixel(a, u, v, c) + get_pixel(b, u, v, c);
                set_pixel(output_image, u, v, c, sum);
            }
        }
    }

    return output_image;
}

image sub_image(image a, image b)
{
    // TODO
    /***********************************************************************
      The input image a and image b have the same height, width, and channels.
      Subtract the given two images and return the result, which should have
      the same height, width, and channels as the inputs. Do necessary checks.
    ************************************************************************/
    assert(a.w == b.w && a.h == b.h && a.c == b.c);
    image output_image = make_image(a.w, a.h, a.c);
    for (int u = 0; u < a.w; ++u) {
        for (int v = 0; v < a.h; ++v) {
            for (int c = 0; c < a.c; ++c) {
                const float sum = get_pixel(a, u, v, c) - get_pixel(b, u, v, c);
                set_pixel(output_image, u, v, c, sum);
            }
        }
    }

    return output_image;
}

image make_gx_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 Sobel Gx filter and return it
    ************************************************************************/
    image gx = make_image(3, 3, 1);
    const float gx_data[] = {-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0};
    for (int i = 0; i < 9; ++i) {
        gx.data[i] = gx_data[i];
    }

    return gx;
}

image make_gy_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 Sobel Gy filter and return it
    ************************************************************************/
    image gy = make_image(3, 3, 1);
    const float gy_data[] = {-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0};
    for (int i = 0; i < 9; ++i) {
        gy.data[i] = gy_data[i];
    }

    return gy;
}

void feature_normalize(image im)
{
    // TODO
    /***********************************************************************
      Calculate minimum and maximum pixel values. Normalize the image by
      subtracting the minimum and dividing by the max-min difference.
    ************************************************************************/
    float maximum = FLT_MIN;
    float minimum = FLT_MAX;
    // Get the min and max
    for (int u = 0; u < im.w; ++u) {
        for (int v = 0; v < im.h; ++v) {
            for (int c = 0; c < im.c; ++c) {
                const float px = get_pixel(im, u, v, c);
                if (px < minimum) { minimum = px;}
                if (px > maximum) { maximum = px;}
            }
        }
    }

    float range = maximum - minimum;
    // Get the min and max
    for (int u = 0; u < im.w; ++u) {
        for (int v = 0; v < im.h; ++v) {
            for (int c = 0; c < im.c; ++c) {
                const float px = get_pixel(im, u, v, c);
                float output_px = 0.0;
                if (range != 0.0) {
                    output_px = (px - minimum) / range;
                }
                set_pixel(im, u, v, c, output_px);
            }
        }
    }
}

image *sobel_image(image im)
{
    // TODO
    /***********************************************************************
      Apply Sobel filter to the input image "im", get the magnitude as sobelimg[0]
      and gradient as sobelimg[1], and return the result.
    ************************************************************************/
    image *sobelimg = calloc(2, sizeof(image));

    const image gx_filter = make_gx_filter();
    const image gy_filter = make_gy_filter();

    const image gx = convolve_image(im, gx_filter, 0); 
    const image gy = convolve_image(im, gy_filter, 0);

    image mag_image = make_image(im.w, im.h, 1);
    for (int u = 0; u < im.w; ++u) {
        for (int v = 0; v < im.h; ++v) {
            const float magnitude = sqrt(pow(get_pixel(gx, u, v, 0), 2) +
                                         pow(get_pixel(gy, u, v, 0), 2));
            set_pixel(mag_image, u, v, 0, magnitude);
        }
    }


    image grad_image = make_image(im.w, im.h, 1);
    for (int u = 0; u < im.w; ++u) {
        for (int v = 0; v < im.h; ++v) {
            const float gradient = atan2(get_pixel(gy, u, v, 0),
                                         get_pixel(gx, u, v, 0));
            set_pixel(grad_image, u, v, 0, gradient);
        }
    }

    sobelimg[0] = mag_image;
    sobelimg[1] = grad_image;

    return sobelimg;
}

image colorize_sobel(image im)
{
    // TODO
    /***********************************************************************
    Create a colorized version of the edges in image "im" using the 
    algorithm described in the README.
    ************************************************************************/
    const image gaussian_filter = make_gaussian_filter(4);
    image smoothed_image = convolve_image(im, gaussian_filter, 1);

    const image* sobelimg = sobel_image(smoothed_image);
    image colorized_sobel_image = make_image(im.w, im.h, 3);

    const image mag_image = sobelimg[0];
    const image grad_image = sobelimg[1];


    feature_normalize(mag_image);
    feature_normalize(grad_image);

    for (int u = 0; u < im.w; ++u) {
        for (int v = 0; v < im.h; ++v) {
            const float grad_pixel = get_pixel(grad_image, u, v, 0);
            set_pixel(colorized_sobel_image, u, v, 0, grad_pixel);

            const float mag_pixel = get_pixel(mag_image, u, v, 0);
            set_pixel(colorized_sobel_image, u, v, 1, mag_pixel);
            set_pixel(colorized_sobel_image, u, v, 2, mag_pixel);
        }
    }

    hsv_to_rgb(colorized_sobel_image);

    return colorized_sobel_image;
}

// EXTRA CREDIT: Median filter

/*
image apply_median_filter(image im, int kernel_size)
{
  return make_image(1,1,1);
}
*/

// SUPER EXTRA CREDIT: Bilateral filter

/*
image apply_bilateral_filter(image im, float sigma1, float sigma2)
{
  return make_image(1,1,1);
}
*/
