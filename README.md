# irANS
A simple zero-order interlaced rANS codec with OpenCL support

## Features

- Support for running on a specific OpenCL device
- Multiblob support for reduced memory usage
- Support for compressed backups
- Detailed verbose output

## Steps to use

- Clone the repository with submodules using `git clone --recurse-submodules -j8 https://github.com/supercmmetry/irans`
- Change working directory using `cd irans`
- Build **irans** using `cmake -DCMAKE_BUILD_TYPE=RELEASE . && make irans`
- Run `irans --help` from the `bin` directory for more details
