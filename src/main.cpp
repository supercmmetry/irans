#include <iostream>
#include <argparse/argparse.h>
#include <rainman/rainman.h>
#include <opencl/cl_helper.h>
#include <multiblob.h>

int main(int argc, const char *argv[]) {
    argparse::ArgumentParser parser(
            "irans",
            "An OpenCL implementation of rANS Codec with zero-order context"
    );

    parser.add_argument()
            .names({"-v", "--verbose"})
            .description("Enable verbose mode")
            .required(false);

    parser.add_argument()
            .names({"-x", "--executor"})
            .description("Executor to use for codec operations (cpu/gpu)")
            .required(false);

    parser.add_argument()
            .names({"-P", "--preferreddevice"})
            .description("Preferred OpenCL device to use for codec operations."
                         " Note that this must match the OpenCL device name of the given hardware.")
            .required(false);

    parser.add_argument()
            .names({"-j", "--jobs"})
            .description("Number of jobs/kernels to run in parallel")
            .required(false);

    parser.add_argument()
            .names({"-b", "--blobsize"})
            .description("Blob size for codec operations")
            .required(false);

    parser.add_argument()
            .names({"-M", "--maxmemory"})
            .description("Set host memory-usage limit")
            .required(false);

    parser.add_argument()
            .names({"-i", "--input"})
            .description("Path for input file")
            .required(false);

    parser.add_argument()
            .names({"-o", "--output"})
            .description("Path for output file")
            .required(false);

    parser.add_argument()
            .names({"-m", "--mode"})
            .description("Mode of operation (c for compression, d for decompression)")
            .required(false);

    parser.add_argument()
            .names({"-l", "--listdevices"})
            .description("List all available OpenCL devices")
            .required(false);

    parser.enable_help();

    auto err = parser.parse(argc, argv);
    if (err) {
        std::cerr << err << std::endl;
        return 1;
    }

    if (parser.exists("help")) {
        parser.print_help();
        return 0;
    }

    if (parser.exists("l")) {
        interlaced_ans::opencl::DeviceProvider::list_available_devices();
        return 0;
    }

    bool verbose = parser.exists("v");
    std::string executor = "cpu";
    std::string mode = "x";
    std::string input_file;
    std::string output_file;
    std::string preferred_device;
    uint64_t jobs = 64;
    uint64_t blob_size = 104857600;
    uint64_t max_mem = 1073741824;

    if (parser.exists("x")) {
        executor = parser.get<std::string>("x");
    }
    if (parser.exists("m")) {
        mode = parser.get<std::string>("m");
    }
    if (parser.exists("i")) {
        input_file = parser.get<std::string>("i");
    }
    if (parser.exists("o")) {
        output_file = parser.get<std::string>("o");
    }
    if (parser.exists("P")) {
        preferred_device = parser.get<std::string>("P");
    }
    if (parser.exists("j")) {
        jobs = parser.get<uint64_t>("j");
    }
    if (parser.exists("b")) {
        blob_size = parser.get<uint64_t>("b");
    }
    if (parser.exists("M")) {
        max_mem = parser.get<uint64_t>("M");
    }

    // Set memory limit on host-machine
    rainman::Allocator().peak_size(max_mem);

    if (executor == "cpu") {
        interlaced_ans::opencl::DeviceProvider::load_devices<CL_DEVICE_TYPE_CPU>();
    } else if (executor == "gpu") {
        interlaced_ans::opencl::DeviceProvider::load_devices<CL_DEVICE_TYPE_GPU>();
    } else {
        interlaced_ans::opencl::DeviceProvider::load_devices<CL_DEVICE_TYPE_ALL>();
    }

    // Set opencl preferred device.
    interlaced_ans::opencl::DeviceProvider::set_preferred_device(preferred_device);

    auto codec = interlaced_ans::MultiBlobCodec(jobs, blob_size, verbose);

    if (input_file.empty()) {
        std::cerr << "Source file not provided" << std::endl;
        return 1;
    }

    if (output_file.empty()) {
        std::cerr << "Destination file not provided" << std::endl;
        return 1;
    }

    if (mode == "c") {
        codec.compress_file(input_file, output_file);
    } else if (mode == "d") {
        codec.decompress_file(input_file, output_file);
    } else {
        std::cerr << "Invalid mode. Choose either 'c' for compression or 'd' for decompression." << std::endl;
        return 1;
    }
}
