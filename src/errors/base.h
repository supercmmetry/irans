#ifndef INTERLACED_ANS_ERRORS_BASE_H
#define INTERLACED_ANS_ERRORS_BASE_H

namespace BaseErrors {
    class InvalidOperationException : public std::exception {
    public:
        std::string msg;

        InvalidOperationException(const std::string &msg) {
            this->msg = "interlaced_ans.base: Invalid operation: " + msg;
        }

        [[nodiscard]] const char *what() const noexcept override {
            return msg.c_str();
        }
    };
}


#endif
