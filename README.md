z.lib
======

[![Join the chat at https://gitter.im/sekrit-twc/zimg](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/sekrit-twc/zimg?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
The "z" library implements the commonly required image processing basics of
scaling, colorspace conversion, and depth conversion. A simple API enables
conversion between any supported formats to operate with minimal knowledge
from the programmer. All library routines were designed from the ground-up
with correctness, flexibility, and thread-safety as first priorities.
Allocation, buffering, and I/O are cleanly separated from processing, allowing
the programmer to adapt "z" to many scenarios.

Requirements
-----
- Byte-addressable architecture
- Two's complement integer encoding
- 32-bit or greater machine word
- C++11 compliant compiler
- Platforms: Microsoft Windows, POSIX

Building
-----
The officially supported build system is GNU autotools. Use the provided
"autogen.sh" script to instantiate the familiar "configure" and "make" build
system. Visual Studio project files are not stable and are subject to change.

Capabilities
-----
### Colorspace

Colorspaces: SMPTE-C (NTSC), Rec.709, Rec.2020

The colorspace module provides for conversion between any combination of
colorspaces, as defined by the commonly used triplet of matrix coefficients,
transfer characteristics, and color primaries. Conversions are implemented
with intelligent logic that minimizes the number of intermediate
representations required for common scenarios, such as conversion between
YCbCr and RGB. Support is also provided for the non-traditional YCbCr system
of ITU-R BT.2020 constant luminance (CL), which retains higher fidelity with
chroma subsampling. Note that "z" is not a color management system and should
not be used to perform drastic contrast or gamut reduction, such as BT.2020
to BT.709.

### Depth

Formats: BYTE, WORD, HALF, FLOAT

The depth module provides for conversion between any pixel (number) format,
including one and two-byte integer formats as well as IEEE-754 binary16
(OpenEXR) and binary32 formats. Limited range (16-235) and full swing (0-255)
integer formats are supported, including conversion between such formats.
Multiple dithering methods are available when converting to integer formats,
from basic rounding to high quality error diffusion.

### Resize

The resize module provides high fidelity linear resamplers, including the
popular Bicubic and Lanczos filters. Resampling ratios of up to 100x are
supported for upsampling and downsampling. Full support is provided for
various coordinate systems, including the various chroma siting conventions
(e.g. JPEG and MPEG2) as well as interlaced images.
