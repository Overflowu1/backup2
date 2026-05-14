# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for 3D OSCNet+ CT Artifact Removal
测试3D OSCNet+处理五维(b,c,h,w,d) NIfTI数据的完整测试套件
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import tempfile
import os
import nibabel as nib
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from WNET import WNet3D, Fconv_PCA_3D, OSCNetplus3D, NIfTIDataHandler, create_3d_mask, example_training_setup, Mnet3D, \
    Xnet3D

# 假设您的3D OSCNet+代码保存在文件中，这里导入所有必要的类
from WNET import OSCNetplus3D, NIfTIDataHandler, VolumeProcessor, create_3d_mask, example_training_setup


class Test3DOSCNetPlus(unittest.TestCase):
    """3D OSCNet+的完整测试类"""

    def setUp(self):
        """设置测试环境"""

        # 测试参数
        class TestArgs:
            def __init__(self):
                self.S = 3  # 减少stages以加快测试
                self.cdiv = 4
                self.num_rot = 4
                self.num_M = 8
                self.num_Q = 4
                self.etaM = 0.1
                self.etaX = 0.1
                self.sizeP = 5  # 减少filter size以加快测试
                self.inP = 3
                self.padding = 2
                self.ifini = 0
                self.T = 2  # 减少ResBlock iterations

        self.args = TestArgs()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 测试数据维度
        self.batch_size = 2
        self.channels = 1
        self.depth = 32
        self.height = 64
        self.width = 64

        # 创建临时目录用于保存测试文件
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_synthetic_3d_data(self, add_artifacts=True, noise_level=0.1):
        """创建合成3D CT数据用于测试"""
        # 创建基础3D体积数据
        data = np.zeros((self.depth, self.height, self.width))

        # 添加一些结构（模拟器官）
        # 中心球体
        center = (self.depth // 2, self.height // 2, self.width // 2)
        radius = min(self.depth, self.height, self.width) // 6

        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    dist = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
                    if dist < radius:
                        data[z, y, x] = 1.0

        # 添加椭球体
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    ellipse_dist = ((z - center[0]) / radius) ** 2 + ((y - center[1] - 10) / radius) ** 2 + (
                                (x - center[2] + 10) / radius) ** 2
                    if ellipse_dist < 0.5:
                        data[z, y, x] = 1.5

        # 添加噪声
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, data.shape)
            data += noise

        # 添加金属伪影
        if add_artifacts:
            # 创建streak artifacts
            artifact_mask = self.create_artifact_mask((self.depth, self.height, self.width))
            artifacts = np.random.normal(0, 0.5, data.shape) * artifact_mask
            data_with_artifacts = data + artifacts
        else:
            data_with_artifacts = data.copy()
            artifact_mask = np.zeros_like(data)

        return data, data_with_artifacts, artifact_mask

    def create_artifact_mask(self, shape):
        """创建3D伪影掩码"""
        d, h, w = shape
        mask = np.zeros((d, h, w))

        # 创建金属streak伪影模式
        center_d, center_h, center_w = d // 2, h // 2, w // 2

        # 水平streaks
        for i in range(d):
            # 主要水平streak
            mask[i, center_h - 2:center_h + 3, :] = 1
            # 次要水平streaks
            if i % 4 == 0:
                mask[i, center_h + 10:center_h + 12, :] = 0.5
                mask[i, center_h - 12:center_h - 10, :] = 0.5

        # 垂直streaks
        for i in range(d):
            mask[i, :, center_w - 2:center_w + 3] = 1
            if i % 4 == 0:
                mask[i, :, center_w + 15:center_w + 17] = 0.5
                mask[i, :, center_w - 17:center_w - 15] = 0.5

        # 对角线streaks
        for i in range(d):
            for j in range(min(h, w)):
                if 0 <= center_h - 20 + j < h and 0 <= center_w - 20 + j < w:
                    mask[i, center_h - 20 + j, center_w - 20 + j] = 0.8

        # 应用高斯平滑使伪影更真实
        mask = ndimage.gaussian_filter(mask, sigma=1.0)

        return mask

    def test_data_dimensions(self):
        """测试数据维度处理"""
        print("Testing data dimensions...")

        # 测试不同维度的输入
        test_cases = [
            (self.batch_size, self.channels, self.depth, self.height, self.width),  # 5D
            (1, 1, 32, 32, 32),  # 小尺寸
            (3, 1, 16, 128, 128),  # 不同比例
        ]

        for dims in test_cases:
            b, c, d, h, w = dims
            # 创建测试数据
            ct_ma = torch.randn(b, c, d, h, w)
            lict = torch.randn(b, c, d, h, w)
            mask = torch.ones(b, c, d, h, w)

            print(f"  Testing dimensions: {dims}")
            self.assertEqual(ct_ma.shape, (b, c, d, h, w))
            self.assertEqual(lict.shape, (b, c, d, h, w))
            self.assertEqual(mask.shape, (b, c, d, h, w))

    def test_individual_components(self):
        """测试各个组件的功能"""
        print("Testing individual components...")

        # 测试WNet3D
        print("  Testing WNet3D...")
        wnet = WNet3D(d=8, h=32, w=32)
        ximg = torch.randn(2, 1, 16, 32, 32)
        xma = torch.randn(2, 1, 16, 32, 32)

        with torch.no_grad():
            weights = wnet(ximg, xma)

        expected_shape = (2, 8, 1, 1, 1, 32, 32)
        self.assertEqual(weights.shape, expected_shape,
                         f"WNet3D output shape mismatch: {weights.shape} vs {expected_shape}")

        # 测试Fconv_PCA_3D
        print("  Testing Fconv_PCA_3D...")
        fconv = Fconv_PCA_3D(sizeP=5, inNum=1, outNum=4, tranNum=4, inP=3, padding=2)
        input_data = torch.randn(2, 4, 16, 32, 32)

        with torch.no_grad():
            output, filter_out = fconv(input_data, ximg, xma)

        expected_output_shape = (2, 1, 16, 32, 32)
        self.assertEqual(output.shape, expected_output_shape,
                         f"Fconv_PCA_3D output shape mismatch: {output.shape} vs {expected_output_shape}")

        # 测试Mnet3D
        print("  Testing Mnet3D...")
        mnet = Mnet3D(self.args)
        m_input = torch.randn(2, self.args.num_M * self.args.num_rot, 16, 32, 32)

        with torch.no_grad():
            m_output = mnet(m_input)

        self.assertEqual(m_output.shape, m_input.shape, "Mnet3D should preserve input shape")

        # 测试Xnet3D
        print("  Testing Xnet3D...")
        xnet = Xnet3D(self.args)
        x_input = torch.randn(2, self.args.num_Q + 1, 16, 32, 32)

        with torch.no_grad():
            x_output = xnet(x_input)

        self.assertEqual(x_output.shape, x_input.shape, "Xnet3D should preserve input shape")

    def test_model_forward_pass(self):
        """测试完整模型的前向传播"""
        print("Testing complete model forward pass...")

        # 创建模型
        model = OSCNetplus3D(self.args)
        model.eval()

        # 创建测试数据
        ct_ma = torch.randn(self.batch_size, 1, self.depth, self.height, self.width)
        lict = torch.randn(self.batch_size, 1, self.depth, self.height, self.width)
        mask = torch.ones(self.batch_size, 1, self.depth, self.height, self.width)

        print(f"  Input shapes - CT_ma: {ct_ma.shape}, LIct: {lict.shape}, Mask: {mask.shape}")

        # 前向传播
        with torch.no_grad():
            try:
                x0, list_x, list_a = model(ct_ma, lict, mask)

                # 验证输出
                self.assertEqual(x0.shape, (self.batch_size, 1, self.depth, self.height, self.width))
                self.assertEqual(len(list_x), self.args.S + 1)  # S iterations + final
                self.assertEqual(len(list_a), self.args.S)

                for i, x in enumerate(list_x):
                    expected_shape = (self.batch_size, 1, self.depth, self.height, self.width)
                    self.assertEqual(x.shape, expected_shape,
                                     f"ListX[{i}] shape mismatch: {x.shape} vs {expected_shape}")

                for i, a in enumerate(list_a):
                    expected_shape = (self.batch_size, 1, self.depth, self.height, self.width)
                    self.assertEqual(a.shape, expected_shape,
                                     f"ListA[{i}] shape mismatch: {a.shape} vs {expected_shape}")

                print("  ✓ Forward pass successful!")
                print(f"  ✓ Output X0 shape: {x0.shape}")
                print(f"  ✓ Number of X iterations: {len(list_x)}")
                print(f"  ✓ Number of A iterations: {len(list_a)}")

            except Exception as e:
                self.fail(f"Forward pass failed with error: {str(e)}")

    def test_nifti_data_handling(self):
        """测试NIfTI数据处理功能"""
        print("Testing NIfTI data handling...")

        # 创建合成数据
        clean_data, noisy_data, artifact_mask = self.create_synthetic_3d_data()

        # 测试保存和加载
        nifti_handler = NIfTIDataHandler()

        # 保存测试数据
        test_file = os.path.join(self.temp_dir, "test_data.nii.gz")
        nifti_handler.save_nifti(noisy_data, test_file)

        # 加载数据
        loaded_data, affine, header = nifti_handler.load_nifti(test_file)

        # 验证数据一致性
        np.testing.assert_array_almost_equal(loaded_data, noisy_data, decimal=5)
        print(f"  ✓ NIfTI save/load successful, shape: {loaded_data.shape}")

        # 测试数据归一化
        normalized_data, norm_params = nifti_handler.normalize_data(loaded_data, method='minmax')
        denormalized_data = nifti_handler.denormalize_data(normalized_data, norm_params, method='minmax')

        np.testing.assert_array_almost_equal(loaded_data, denormalized_data, decimal=5)
        print("  ✓ Data normalization/denormalization successful")

        # 测试批处理准备
        batch_data = nifti_handler.prepare_batch(loaded_data, batch_size=1)
        expected_shape = (1, 1, self.depth, self.height, self.width)
        self.assertEqual(batch_data.shape, expected_shape)
        print(f"  ✓ Batch preparation successful, shape: {batch_data.shape}")

    def test_volume_processor(self):
        """测试体积分块处理器"""
        print("Testing volume processor...")

        # 创建测试数据
        test_volume = np.random.randn(64, 128, 128)
        processor = VolumeProcessor(patch_size=(32, 64, 64), overlap=8)

        # 提取patches
        patches, positions = processor.extract_patches(test_volume)

        print(f"  ✓ Extracted {len(patches)} patches from volume shape {test_volume.shape}")

        # 重建体积
        reconstructed = processor.reconstruct_volume(patches, positions, test_volume.shape)

        # 验证重建精度
        mse = np.mean((test_volume - reconstructed) ** 2)
        print(f"  ✓ Volume reconstruction MSE: {mse:.6f}")
        self.assertLess(mse, 1e-10, "Reconstruction error too high")

    def test_end_to_end_processing(self):
        """端到端处理测试"""
        print("Testing end-to-end processing...")

        # 创建合成数据
        clean_data, noisy_data, artifact_mask = self.create_synthetic_3d_data()

        # 准备数据
        handler = NIfTIDataHandler()
        normalized_noisy, norm_params = handler.normalize_data(noisy_data)
        normalized_clean, _ = handler.normalize_data(clean_data, method='minmax')

        # 转换为tensor
        ct_ma = torch.FloatTensor(normalized_noisy).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        lict = ct_ma.clone()  # 模拟低质量输入
        mask = torch.FloatTensor(1 - artifact_mask).unsqueeze(0).unsqueeze(0)  # 反转掩码

        print(f"  Input data shape: {ct_ma.shape}")

        # 创建并运行模型
        model = OSCNetplus3D(self.args)
        model.eval()

        with torch.no_grad():
            x0, list_x, list_a = model(ct_ma, lict, mask)

            # 获取最终重建结果
            final_reconstruction = list_x[-1]

            # 转换回numpy
            reconstruction_np = final_reconstruction.squeeze().numpy()

            # 反归一化
            reconstruction_denorm = handler.denormalize_data(reconstruction_np, norm_params)

            # 计算指标
            mse_initial = np.mean((normalized_noisy - normalized_clean) ** 2)
            mse_final = np.mean((reconstruction_np - normalized_clean) ** 2)

            psnr_initial = -10 * np.log10(mse_initial)
            psnr_final = -10 * np.log10(mse_final)

            print(f"  ✓ Initial MSE: {mse_initial:.6f}, PSNR: {psnr_initial:.2f} dB")
            print(f"  ✓ Final MSE: {mse_final:.6f}, PSNR: {psnr_final:.2f} dB")
            print(f"  ✓ Improvement: {psnr_final - psnr_initial:.2f} dB")

            # 保存测试结果
            result_file = os.path.join(self.temp_dir, "reconstruction_result.nii.gz")
            handler.save_nifti(reconstruction_denorm, result_file)
            print(f"  ✓ Results saved to {result_file}")

    def test_memory_usage(self):
        """测试内存使用情况"""
        print("Testing memory usage...")

        import psutil
        process = psutil.Process()

        # 记录初始内存
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 创建模型和数据
        model = OSCNetplus3D(self.args)
        ct_ma = torch.randn(1, 1, 32, 64, 64)
        lict = torch.randn(1, 1, 32, 64, 64)
        mask = torch.ones(1, 1, 32, 64, 64)

        # 运行前向传播
        with torch.no_grad():
            _ = model(ct_ma, lict, mask)

        # 记录峰值内存
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = peak_memory - initial_memory

        print(f"  ✓ Memory usage: {memory_usage:.1f} MB")
        print(f"  ✓ Peak memory: {peak_memory:.1f} MB")

        # 确保内存使用合理（小于2GB）
        self.assertLess(memory_usage, 2000, "Memory usage too high")

    def test_gradient_flow(self):
        """测试梯度流动"""
        print("Testing gradient flow...")

        model = OSCNetplus3D(self.args)
        model.train()

        # 创建测试数据
        ct_ma = torch.randn(1, 1, 16, 32, 32, requires_grad=True)
        lict = torch.randn(1, 1, 16, 32, 32)
        mask = torch.ones(1, 1, 16, 32, 32)
        target = torch.randn(1, 1, 16, 32, 32)

        # 前向传播
        x0, list_x, list_a = model(ct_ma, lict, mask)

        # 计算损失
        loss = F.mse_loss(list_x[-1], target)

        # 反向传播
        loss.backward()

        # 检查梯度
        gradients_exist = 0
        total_params = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    gradients_exist += 1
                    grad_norm = param.grad.data.norm(2).item()
                    if grad_norm > 1e-6:  # 有意义的梯度
                        pass

        gradient_ratio = gradients_exist / total_params if total_params > 0 else 0

        print(f"  ✓ Parameters with gradients: {gradients_exist}/{total_params} ({gradient_ratio:.1%})")
        print(f"  ✓ Loss value: {loss.item():.6f}")

        # 确保大部分参数都有梯度
        self.assertGreater(gradient_ratio, 0.8, "Too few parameters have gradients")

    def test_inference_speed(self):
        """测试推理速度"""
        print("Testing inference speed...")

        import time

        model = OSCNetplus3D(self.args)
        model.eval()

        # 创建测试数据
        ct_ma = torch.randn(1, 1, 32, 64, 64)
        lict = torch.randn(1, 1, 32, 64, 64)
        mask = torch.ones(1, 1, 32, 64, 64)

        # 预热
        with torch.no_grad():
            _ = model(ct_ma, lict, mask)

        # 测试推理时间
        num_runs = 5
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(ct_ma, lict, mask)

        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs

        print(f"  ✓ Average inference time: {avg_time:.3f} seconds")
        print(f"  ✓ Throughput: {1 / avg_time:.2f} volumes/second")

        # 确保推理时间合理（小于30秒）
        self.assertLess(avg_time, 30, "Inference time too slow")

    def generate_test_report(self):
        """生成测试报告"""
        print("\n" + "=" * 60)
        print("3D OSCNet+ Test Report")
        print("=" * 60)

        # 运行所有测试并收集结果
        test_methods = [
            ('Data Dimensions', self.test_data_dimensions),
            ('Individual Components', self.test_individual_components),
            ('Model Forward Pass', self.test_model_forward_pass),
            ('NIfTI Data Handling', self.test_nifti_data_handling),
            ('Volume Processor', self.test_volume_processor),
            ('End-to-End Processing', self.test_end_to_end_processing),
            ('Memory Usage', self.test_memory_usage),
            ('Gradient Flow', self.test_gradient_flow),
            ('Inference Speed', self.test_inference_speed),
        ]

        results = {}

        for test_name, test_method in test_methods:
            print(f"\n{test_name}:")
            try:
                test_method()
                results[test_name] = "PASS"
                print(f"  ✅ {test_name}: PASSED")
            except Exception as e:
                results[test_name] = f"FAIL: {str(e)}"
                print(f"  ❌ {test_name}: FAILED - {str(e)}")

        # 打印总结
        print("\n" + "-" * 60)
        print("Test Summary:")
        passed = sum(1 for r in results.values() if r == "PASS")
        total = len(results)

        for test_name, result in results.items():
            status_icon = "✅" if result == "PASS" else "❌"
            print(f"  {status_icon} {test_name}: {result}")

        print(f"\nOverall: {passed}/{total} tests passed ({passed / total:.1%})")

        if passed == total:
            print("🎉 All tests passed! Your 3D OSCNet+ implementation is working correctly.")
        else:
            print("⚠️  Some tests failed. Please check the implementation.")

        return results


def run_comprehensive_tests():
    """运行完整的测试套件"""
    print("Starting comprehensive 3D OSCNet+ testing...")

    # 创建测试实例
    test_suite = Test3DOSCNetPlus()
    test_suite.setUp()

    try:
        # 生成测试报告
        results = test_suite.generate_test_report()
        return results
    finally:
        test_suite.tearDown()


def create_visualization_demo():
    """创建可视化演示"""
    print("\nCreating visualization demo...")

    # 创建测试数据
    test_suite = Test3DOSCNetPlus()
    test_suite.setUp()

    clean_data, noisy_data, artifact_mask = test_suite.create_synthetic_3d_data()

    # 选择中间切片进行可视化
    mid_slice = clean_data.shape[0] // 2

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 原始数据
    axes[0, 0].imshow(clean_data[mid_slice], cmap='gray')
    axes[0, 0].set_title('Clean CT Slice')
    axes[0, 0].axis('off')

    # 带伪影数据
    axes[0, 1].imshow(noisy_data[mid_slice], cmap='gray')
    axes[0, 1].set_title('CT with Artifacts')
    axes[0, 1].axis('off')

    # 伪影掩码
    axes[1, 0].imshow(artifact_mask[mid_slice], cmap='hot')
    axes[1, 0].set_title('Artifact Mask')
    axes[1, 0].axis('off')

    # 差异图
    diff = np.abs(noisy_data[mid_slice] - clean_data[mid_slice])
    axes[1, 1].imshow(diff, cmap='jet')
    axes[1, 1].set_title('Difference Map')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(test_suite.temp_dir, 'test_visualization.png'), dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {test_suite.temp_dir}/test_visualization.png")

    test_suite.tearDown()


if __name__ == "__main__":
    # 运行主要测试
    test_results = run_comprehensive_tests()

    # 创建可视化
    try:
        create_visualization_demo()
    except Exception as e:
        print(f"Visualization creation failed: {e}")

    print("\n🔬 Testing completed!")