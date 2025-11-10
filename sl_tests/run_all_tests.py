"""
运行所有测试用例

按顺序运行所有测试，逐步验证 Reward-Shifted Speculative Sampling 的实现。
"""
import sys
import traceback

def run_test(test_name, test_func):
    """运行单个测试"""
    print(f"\n{'='*80}")
    print(f"开始运行: {test_name}")
    print('='*80)
    try:
        test_func()
        print(f"\n✓ {test_name} 通过")
        return True
    except Exception as e:
        print(f"\n✗ {test_name} 失败")
        print(f"错误: {e}")
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("="*80)
    print("Reward-Shifted Speculative Sampling 测试套件")
    print("="*80)
    
    # 导入所有测试
    from test_small_base_probs_extraction import test_small_base_probs_extraction
    from test_accept_logic import test_accept_logic_with_small_base
    from test_recover_logic import test_recover_logic
    from test_end_to_end import test_end_to_end
    from test_backward_compatibility import test_backward_compatibility
    
    tests = [
        ("测试 1: small_base_probs 提取", test_small_base_probs_extraction),
        ("测试 2: accept 逻辑", test_accept_logic_with_small_base),
        ("测试 3: recover 逻辑", test_recover_logic),
        ("测试 4: 端到端测试", test_end_to_end),
        ("测试 5: 向后兼容性", test_backward_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        success = run_test(test_name, test_func)
        results.append((test_name, success))
    
    # 打印总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {status}: {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())

