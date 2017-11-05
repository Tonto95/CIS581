clear all
close all

im = imread('test_imgs/test16.JPG');
im = imresize(im, 0.25);

imwrite(permute(im, [2 1 3]), 'test16.JPG');
