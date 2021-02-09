#ifndef INTERLACED_ANS_ERRORS_OPENCL_H
#define INTERLACED_ANS_ERRORS_OPENCL_H

namespace OpenCLErrors {
    class InvalidOperationException : public std::exception {
    public:
        std::string msg;

        InvalidOperationException(const std::string &msg) {
            this->msg = "interlaced_ans.opencl: Invalid operation: " + msg;
        }

        [[nodiscard]] const char *what() const noexcept override {
            return msg.c_str();
        }
    };
}
#endif