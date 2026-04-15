"""
Run all phase tests to verify implementation.

Usage:
    python run_all_tests.py
"""

import subprocess
import sys


def run_test(test_script, phase_name):
    """Run a test script and report results."""
    print("\n" + "="*80)
    print(f"RUNNING: {phase_name}")
    print("="*80)
    
    try:
        result = subprocess.run(
            [sys.executable, test_script],
            capture_output=False,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"\n✅ {phase_name} PASSED")
            return True
        else:
            print(f"\n❌ {phase_name} FAILED (exit code: {result.returncode})")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"\n❌ {phase_name} TIMEOUT")
        return False
    except Exception as e:
        print(f"\n❌ {phase_name} ERROR: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE PHASE TESTING")
    print("="*80)
    print("\nTesting all implemented phases...")
    
    tests = [
        ("test_phase6_attention.py", "Phase 6: Pocket-Guided Attention"),
        ("test_phase7_uncertainty.py", "Phase 7: Evidential Uncertainty"),
        ("test_phase9_multitask.py", "Phase 9: Multi-Task Learning"),
    ]
    
    results = {}
    
    for test_script, phase_name in tests:
        results[phase_name] = run_test(test_script, phase_name)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for phase_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{status} - {phase_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nYour implementation is production-ready!")
        print("\nNext steps:")
        print("  1. Extract embeddings: python extract_embeddings.py")
        print("  2. Generate visualizations: python generate_visualizations.py")
        print("  3. Train with all features enabled")
        print("  4. Run ablation studies")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("\nPlease review the failed tests above.")
        return 1


if __name__ == "__main__":
    exit(main())
