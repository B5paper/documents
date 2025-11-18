#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <ctime>

// 设备函数：生成随机数
struct RandomTransform {
    __host__ __device__
    unsigned int operator()(unsigned int x) const {
        return (x * 1103515245 + 12345) % 1000;  // 简单的伪随机数生成
    }
};

// 使用 Thrust 库实现基数排序（最简单可靠的方法）
void thrustRadixSortDemo() {
    std::cout << "=== Thrust 基数排序演示 ===" << std::endl;
    
    const int N = 20;  // 使用小数据量便于演示
    
    // 在主机上生成随机数据
    std::vector<unsigned int> h_data(N);
    for(int i = 0; i < N; i++) {
        h_data[i] = rand() % 1000;  // 0-999 的随机数
    }
    
    // 将数据拷贝到设备
    thrust::device_vector<unsigned int> d_data = h_data;
    
    std::cout << "排序前: ";
    for(int i = 0; i < N; i++) {
        std::cout << std::setw(4) << h_data[i] << " ";
    }
    std::cout << std::endl;
    
    // 使用 thrust::sort 进行基数排序
    thrust::sort(d_data.begin(), d_data.end());
    
    // 将结果拷贝回主机
    thrust::copy(d_data.begin(), d_data.end(), h_data.begin());
    
    std::cout << "排序后: ";
    for(int i = 0; i < N; i++) {
        std::cout << std::setw(4) << h_data[i] << " ";
    }
    std::cout << std::endl;
    
    // 验证排序结果
    bool is_sorted = thrust::is_sorted(d_data.begin(), d_data.end());
    std::cout << "验证: 数组" << (is_sorted ? "已正确排序" : "排序失败") << std::endl;
    std::cout << std::endl;
}

// 性能测试：大数据量排序
void performanceTest() {
    std::cout << "=== 性能测试 ===" << std::endl;
    
    const int N = 1000000;  // 100万数据
    
    // 在主机生成数据然后拷贝到设备
    std::vector<unsigned int> h_data(N);
    for(int i = 0; i < N; i++) {
        h_data[i] = rand() % 1000000;
    }
    
    thrust::device_vector<unsigned int> d_data = h_data;
    
    std::cout << "数据量: " << N << " 个元素" << std::endl;
    
    // 计时开始
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // 执行排序
    thrust::sort(d_data.begin(), d_data.end());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "排序耗时: " << milliseconds << " 毫秒" << std::endl;
    
    // 验证排序正确性
    bool is_sorted = thrust::is_sorted(d_data.begin(), d_data.end());
    std::cout << "正确性验证: " << (is_sorted ? "✓ 通过" : "✗ 失败") << std::endl;
    
    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout << std::endl;
}

// 键值对排序演示
void keyValueSortDemo() {
    std::cout << "=== 键值对排序演示 ===" << std::endl;
    
    const int N = 10;
    
    // 创建键和值
    thrust::device_vector<int> keys(N);
    thrust::device_vector<int> values(N);
    
    // 初始化随机键值 - 使用 thrust::transform 和函数对象
    thrust::sequence(keys.begin(), keys.end(), 0);
    thrust::transform(keys.begin(), keys.end(), keys.begin(), RandomTransform());
    
    thrust::sequence(values.begin(), values.end(), 100);  // 100,101,102,...109
    
    // 拷贝到主机显示
    std::vector<int> h_keys(N);
    std::vector<int> h_values(N);
    thrust::copy(keys.begin(), keys.end(), h_keys.begin());
    thrust::copy(values.begin(), values.end(), h_values.begin());
    
    std::cout << "排序前:" << std::endl;
    std::cout << "Keys:   ";
    for(int i = 0; i < N; i++) {
        std::cout << std::setw(3) << h_keys[i] << " ";
    }
    std::cout << std::endl << "Values: ";
    for(int i = 0; i < N; i++) {
        std::cout << std::setw(3) << h_values[i] << " ";
    }
    std::cout << std::endl;
    
    // 根据键排序，值会跟随键一起移动
    thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
    
    // 拷贝结果回主机显示
    thrust::copy(keys.begin(), keys.end(), h_keys.begin());
    thrust::copy(values.begin(), values.end(), h_values.begin());
    
    std::cout << "排序后:" << std::endl;
    std::cout << "Keys:   ";
    for(int i = 0; i < N; i++) {
        std::cout << std::setw(3) << h_keys[i] << " ";
    }
    std::cout << std::endl << "Values: ";
    for(int i = 0; i < N; i++) {
        std::cout << std::setw(3) << h_values[i] << " ";
    }
    std::cout << std::endl;
    
    // 验证键是否已排序
    bool keys_sorted = thrust::is_sorted(keys.begin(), keys.end());
    std::cout << "键排序验证: " << (keys_sorted ? "✓ 通过" : "✗ 失败") << std::endl;
    std::cout << std::endl;
}

// 降序排序演示
void descendingSortDemo() {
    std::cout << "=== 降序排序演示 ===" << std::endl;
    
    const int N = 15;
    thrust::device_vector<int> data(N);
    
    // 生成随机数据 - 使用函数对象替代 lambda
    thrust::sequence(data.begin(), data.end(), 0);
    thrust::transform(data.begin(), data.end(), data.begin(), RandomTransform());
    
    // 拷贝到主机显示
    std::vector<int> h_data(N);
    thrust::copy(data.begin(), data.end(), h_data.begin());
    
    std::cout << "原始数据: ";
    for(int i = 0; i < N; i++) {
        std::cout << std::setw(3) << h_data[i] << " ";
    }
    std::cout << std::endl;
    
    // 升序排序
    thrust::sort(data.begin(), data.end());
    thrust::copy(data.begin(), data.end(), h_data.begin());
    std::cout << "升序排序: ";
    for(int i = 0; i < N; i++) {
        std::cout << std::setw(3) << h_data[i] << " ";
    }
    std::cout << std::endl;
    
    // 重新生成数据用于降序排序
    thrust::sequence(data.begin(), data.end(), 0);
    thrust::transform(data.begin(), data.end(), data.begin(), RandomTransform());
    thrust::copy(data.begin(), data.end(), h_data.begin());
    
    std::cout << "原始数据: ";
    for(int i = 0; i < N; i++) {
        std::cout << std::setw(3) << h_data[i] << " ";
    }
    std::cout << std::endl;
    
    // 降序排序（使用 greater<int>() 比较函数）
    thrust::sort(data.begin(), data.end(), thrust::greater<int>());
    thrust::copy(data.begin(), data.end(), h_data.begin());
    std::cout << "降序排序: ";
    for(int i = 0; i < N; i++) {
        std::cout << std::setw(3) << h_data[i] << " ";
    }
    std::cout << std::endl;
    
    // 验证降序排序
    bool is_descending = thrust::is_sorted(data.begin(), data.end(), thrust::greater<int>());
    std::cout << "降序验证: " << (is_descending ? "✓ 通过" : "✗ 失败") << std::endl;
    std::cout << std::endl;
}

// 浮点数排序演示
void floatSortDemo() {
    std::cout << "=== 浮点数排序演示 ===" << std::endl;
    
    const int N = 10;
    thrust::device_vector<float> data(N);
    
    // 生成随机浮点数
    for(int i = 0; i < N; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
    }
    
    std::vector<float> h_data(N);
    thrust::copy(data.begin(), data.end(), h_data.begin());
    
    std::cout << "排序前: ";
    for(int i = 0; i < N; i++) {
        std::cout << std::fixed << std::setprecision(2) << h_data[i] << " ";
    }
    std::cout << std::endl;
    
    // 排序
    thrust::sort(data.begin(), data.end());
    thrust::copy(data.begin(), data.end(), h_data.begin());
    
    std::cout << "排序后: ";
    for(int i = 0; i < N; i++) {
        std::cout << std::fixed << std::setprecision(2) << h_data[i] << " ";
    }
    std::cout << std::endl;
    
    bool is_sorted = thrust::is_sorted(data.begin(), data.end());
    std::cout << "验证: " << (is_sorted ? "✓ 通过" : "✗ 失败") << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "CUDA 基数排序完整演示程序" << std::endl;
    std::cout << "==========================" << std::endl << std::endl;
    
    // 设置随机种子
    srand(static_cast<unsigned int>(time(nullptr)));
    
    // 演示1：基础排序
    thrustRadixSortDemo();
    
    // 演示2：键值对排序
    keyValueSortDemo();
    
    // 演示3：降序排序
    descendingSortDemo();
    
    // 演示4：浮点数排序
    floatSortDemo();
    
    // 演示5：性能测试
    performanceTest();
    
    std::cout << "所有演示完成！" << std::endl;
    
    return 0;
}
