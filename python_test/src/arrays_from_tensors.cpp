#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>

using namespace tensorflow;

template<typename T, int NDIMS>
class TensorConversion {
public:
static T* AccessDataPointer(const tensorflow::Tensor &tensor) {
        auto tensor_map = tensor.tensor<T, NDIMS>();
        auto array = tensor_map.data();

        return const_cast<T*>(array);
}
};

int main(int argc, const char *argv[]) {
        const int batch_size = 1;
        const int depth = 5;
        const int height = 5;
        const int width = 5;
        const int channels = 3;

        Tensor tensor(DataType::DT_INT32, TensorShape({batch_size, depth, height, width, channels}));

        auto tensor_map = tensor.tensor<int, 5>();

        for (size_t n = 0; n < batch_size; n++) {
                for (size_t d = 0; d < depth; d++) {
                        std::cout << d << " --" << '\n';

                        for (size_t h = 0; h < height; h++) {
                                for (size_t w = 0; w < width; w++) {
                                        for (size_t c = 0; c < channels; c++) {
                                                tensor_map(n, d, h, w, c) = (((n * depth + d) * height + h) * width + w) * channels + c;
                                                std::cout << tensor_map(n, d, h, w, c) << ",";
                                        }
                                        std::cout << " ";
                                }
                                std::cout << '\n';
                        }
                }
        }

        auto array = tensor_map.data();
        int *int_array = static_cast<int *>(array);

        for (size_t n = 0; n < batch_size; n++) {
                for (size_t d = 0; d < depth; d++) {
                        std::cout << d << " --" << std::endl;

                        for (size_t h = 0; h < height; h++) {
                                for (size_t w = 0; w < width; w++) {
                                        for (size_t c = 0; c < channels; c++) {
                                                std::cout << int_array[(((n * depth + d) * height + h) * width + w) * channels + c] << ',';
                                        }
                                        std::cout << " ";
                                }
                                std::cout << '\n';
                        }
                }
        }

        return 0;
}
