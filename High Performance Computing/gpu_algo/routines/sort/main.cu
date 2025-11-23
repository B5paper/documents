#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iomanip>

// 生成随机数函数
struct RandomNumber {
    float operator()() {
        return static_cast<float>(rand()) / RAND_MAX * 1000.0f;
    }
};

int main() {
    // 设置随机数种子
    srand(static_cast<unsigned>(time(nullptr)));
    
    const size_t N = 20;  // 数据量大小
    
    std::cout << "CUDA Thrust 排序示例" << std::endl;
    std::cout << "数据量: " << N << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    // 1. 在主机上创建并初始化数据
    thrust::host_vector<float> h_data(N);
    thrust::generate(h_data.begin(), h_data.end(), RandomNumber());
    
    std::cout << "排序前数据:" << std::endl;
    for(size_t i = 0; i < h_data.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2) << h_data[i] << " ";
        if((i + 1) % 10 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
    
    // 2. 将数据传输到设备
    thrust::device_vector<float> d_data = h_data;
    
    // 3. 在GPU上进行排序
    thrust::sort(d_data.begin(), d_data.end());
    
    // 4. 将结果传回主机
    thrust::copy(d_data.begin(), d_data.end(), h_data.begin());
    
    std::cout << "排序后数据:" << std::endl;
    for(size_t i = 0; i < h_data.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2) << h_data[i] << " ";
        if((i + 1) % 10 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
    
    // 验证排序结果
    bool is_sorted = true;
    for(size_t i = 1; i < h_data.size(); ++i) {
        if(h_data[i] < h_data[i-1]) {
            is_sorted = false;
            break;
        }
    }
    
    if(is_sorted) {
        std::cout << "✓ 排序验证成功: 数据已正确排序" << std::endl;
    } else {
        std::cout << "✗ 排序验证失败" << std::endl;
    }
    
    return 0;
}
