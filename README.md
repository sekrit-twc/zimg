z.lib
======
The "z" library implements the commonly required image processing basics of scaling, color conversion, and depth conversion. Each basic function is exposed in a simple C API designed to be adaptable to any user scenario. Allocation, buffering, and other such details are cleanly separated from the image processing, allowing the user to implement such concerns to best fit his use case.

Requirements
-----
- Byte-addressable architecture
- Two's complement integer encoding
- 32-bit or greater machine word
- Platforms: Microsoft Windows, POSIX

Building
-----
The officially supported build system is GNU autotools. Use the provided "autogen.sh" script to instantiate the familiar "configure" and "make" build system.

Capabilities
-----
###Colorspace
Supported formats: HALF, FLOAT

Colorspaces: SMPTE-C (NTSC), Rec.709, Rec.2020

The colorspace module provides support for conversion between any combination of matrix coefficients, transfer characteristics, and color primaries. This includes the basic conversions between YCbCr and RGB, as well as more advanced operations such as conversion between linear and gamma encoding and widening from Rec.709 to Rec.2020 gamut. Full support is also provided for constant-luminance (CL) Rec.2020 YCbCr, which retains higher fidelity in chroma subsampling scenarios. However, color management (CMS) is not included in the module, so narrowing conversions, such as from Rec.2020 to Rec.709 may produce suboptimal results.

###Depth
Supported formats: BYTE, WORD, HALF, FLOAT

The depth module provides support for converting between any pixel (number) format, including single and dual-byte integer formats as well as IEEE-754 binary16 and binary32 formats. Both limited (studio) and full (PC) range integer formats are supported, including conversion in either direction. When converting to an integral format, multiple dithering methods are available, including rounding, bayer (ordered) dithering, random dithering, and Floyd-Steinberg error diffusion.

###Resize
Supported formats: WORD, HALF, FLOAT

The resize module provides high fidelity linear resamplers, such as the popular Bicubic and Lanczos filters. Resampling ratios up to 100x are supported without issue for both upsampling and downsampling. Full support is also provided for sub-pixel center shifts and cropping, allowing conversion between different coordinate systems, such as JPEG and MPEG-2 chroma siting.
