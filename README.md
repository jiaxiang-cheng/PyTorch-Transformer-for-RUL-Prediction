# PyTorch Transformer for RUL Prediction
An implementation with Transformer encoder and convolution layers with PyTorch for remaining useful life prediction.

## Quick Run
Simply run `pyhton train.py`. And you will get the training loss and testing result for each epoch:
```
Epoch: 0, loss: 9474.43470, RMSE: 61.11946
Epoch: 1, loss: 5858.27227, RMSE: 46.03318
Epoch: 2, loss: 3208.53410, RMSE: 29.78244
Epoch: 3, loss: 1310.71390, RMSE: 22.94705
...
```
The testing is conducted for each epoch as the data set is not large so it's no big deal but you may remove them and only do the evaluation after finishing the training epochs.

## Prediction Results
The current model can achieve root mean square error in a range of 20 to 40 with random seeds, which is still not far from the SOTA. One of the best testing results achieved on FD001 is shown as follows:
![](https://github.com/jiaxiang-cheng/transformer-pytorch-remaining-useful-life-prediction/blob/main/best%20trial%20with%20FD001.png?raw=true)

![](https://github.com/jiaxiang-cheng/transformer-pytorch-remaining-useful-life-prediction/blob/main/example%20with%20unit%2043%20in%20FD001.png?raw=true)

You may also visualize the testing result for a single unit along the timeline. Here's the visualization of unit 43 in FD001 above. Look forward to help with setting proper training modes and hyperparameters.
## Environment Details
Here follows the details with `conda list`:
```
# Name                    Version                   Build  Channel
blas                      1.0                         mkl  
bzip2                     1.0.8                h1de35cc_0  
ca-certificates           2021.4.13            hecd8cb5_1  
certifi                   2020.12.5        py38hecd8cb5_0  
cycler                    0.10.0                   py38_0  
dill                      0.3.3              pyhd3eb1b0_0  
ffmpeg                    4.3                  h0a44026_0    pytorch
freetype                  2.10.4               ha233b18_0  
gettext                   0.21.0               h7535e17_0  
gmp                       6.2.1                h23ab428_2  
gnutls                    3.6.15               hed9c0bf_0  
icu                       58.2                 h0a44026_3  
intel-openmp              2021.2.0           hecd8cb5_564  
jpeg                      9b                   he5867d9_2  
kiwisolver                1.3.1            py38h23ab428_0  
lame                      3.100                h1de35cc_0  
lcms2                     2.12                 hf1fd2bf_0  
libcxx                    10.0.0                        1  
libffi                    3.3                  hb1e8313_2  
libgfortran               3.0.1                h93005f0_2  
libiconv                  1.16                 h1de35cc_0  
libidn2                   2.3.0                h9ed2024_0  
libpng                    1.6.37               ha441bb4_0  
libtasn1                  4.16.0               h9ed2024_0  
libtiff                   4.1.0                hcb84e12_1  
libunistring              0.9.10               h9ed2024_0  
libuv                     1.40.0               haf1e3a3_0  
libxml2                   2.9.10               h7cdb67c_3  
llvm-openmp               10.0.0               h28b9765_0  
lz4-c                     1.9.3                h23ab428_0  
matplotlib                3.3.4            py38hecd8cb5_0  
matplotlib-base           3.3.4            py38h8b3ea08_0  
mkl                       2021.2.0           hecd8cb5_269  
mkl-service               2.3.0            py38h9ed2024_1  
mkl_fft                   1.3.0            py38h4a7008c_2  
mkl_random                1.2.1            py38hb2f4e1b_2  
ncurses                   6.2                  h0a44026_1  
nettle                    3.7.2                h230ac6f_1  
ninja                     1.10.2               hf7b0b51_1  
numpy                     1.20.1           py38hd6e1bb9_0  
numpy-base                1.20.1           py38h585ceec_0  
olefile                   0.46                       py_0  
openh264                  2.1.0                hd9629dc_0  
openssl                   1.1.1k               h9ed2024_0  
pandas                    1.2.4            py38h23ab428_0  
pillow                    8.2.0            py38h5270095_0  
pip                       21.0.1           py38hecd8cb5_0  
pyparsing                 2.4.7              pyhd3eb1b0_0  
python                    3.8.8                h88f2d9e_5  
python-dateutil           2.8.1              pyhd3eb1b0_0  
pytorch                   1.8.1                   py3.8_0    pytorch
pytz                      2021.1             pyhd3eb1b0_0  
readline                  8.1                  h9ed2024_0  
scipy                     1.6.2            py38hd5f7400_1  
seaborn                   0.11.1             pyhd3eb1b0_0  
setuptools                52.0.0           py38hecd8cb5_0  
six                       1.15.0           py38hecd8cb5_0  
sqlite                    3.35.4               hce871da_0  
tk                        8.6.10               hb0a8c7a_0  
torchvision               0.9.1                  py38_cpu    pytorch
tornado                   6.1              py38h9ed2024_0  
typing_extensions         3.7.4.3            pyha847dfd_0  
wheel                     0.36.2             pyhd3eb1b0_0  
xz                        5.2.5                h1de35cc_0  
zlib                      1.2.11               h1de35cc_3  
zstd                      1.4.9                h322a384_0  
```

## Credit
This work is inpired by Mo, Y., Wu, Q., Li, X., & Huang, B. (2021). Remaining useful life estimation via transformer encoder enhanced by a gated convolutional unit. Journal of Intelligent Manufacturing, 1-10.
