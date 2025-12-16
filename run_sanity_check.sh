#!/bin/bash
# Comprehensive sanity check for all ex02 functionality

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "COMPREHENSIVE SANITY CHECK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

CONDA_ENV="/proj/aimi-adl/envs/adl23_2"
PASSED=0
FAILED=0

# Test 1: Plot Beta Schedules
echo "Test 1: Plot Beta Schedules"
echo "  Command: python ex02_main.py --plot_schedules"
if conda run -p $CONDA_ENV python ex02_main.py --plot_schedules > /dev/null 2>&1; then
    if [ -f "visualisations/beta_schedules_comparison.png" ]; then
        echo "  ✓ PASSED"
        ((PASSED++))
    else
        echo "  ✗ FAILED (output file not created)"
        ((FAILED++))
    fi
else
    echo "  ✗ FAILED (execution error)"
    ((FAILED++))
fi
echo ""

# Test 2: Calculate FID
echo "Test 2: Calculate FID Score"
echo "  Command: python ex02_main.py --calculate_fid --generated_dir outputs_conditional_10ep"
if conda run -p $CONDA_ENV python ex02_main.py --calculate_fid --generated_dir outputs_conditional_10ep > /dev/null 2>&1; then
    if [ -f "outputs_conditional_10ep/fid_score.txt" ]; then
        echo "  ✓ PASSED"
        echo "    FID: $(grep 'FID Score' outputs_conditional_10ep/fid_score.txt)"
        ((PASSED++))
    else
        echo "  ✗ FAILED (fid_score.txt not created)"
        ((FAILED++))
    fi
else
    echo "  ✗ FAILED (execution error)"
    ((FAILED++))
fi
echo ""

# Test 3: Inference (Unconditional)
echo "Test 3: Inference Mode (Unconditional)"
echo "  Command: python ex02_main.py --inference --model_path models/DDPM_30epochs_cosine/ckpt.pt --num_samples 4"
rm -rf test_sanity_inference
if conda run -p $CONDA_ENV python ex02_main.py --inference --model_path models/DDPM_30epochs_cosine/ckpt.pt --num_samples 4 --save_dir test_sanity_inference > /dev/null 2>&1; then
    IMG_COUNT=$(find test_sanity_inference -name "*.png" | wc -l)
    if [ "$IMG_COUNT" -eq 4 ]; then
        echo "  ✓ PASSED (4 images generated)"
        ((PASSED++))
    else
        echo "  ✗ FAILED (expected 4 images, got $IMG_COUNT)"
        ((FAILED++))
    fi
else
    echo "  ✗ FAILED (execution error)"
    ((FAILED++))
fi
echo ""

# Test 4: Create Animation (Forward)
echo "Test 4: Create Animation (Forward Process)"
echo "  Command: python create_animation.py --model_path models/DDPM_30epochs_cosine/ckpt.pt --mode forward --frame_interval 10"
rm -f test_forward.gif
if timeout 60 conda run -p $CONDA_ENV python create_animation.py --model_path models/DDPM_30epochs_cosine/ckpt.pt --mode forward --frame_interval 10 --duration 100 > /dev/null 2>&1; then
    if [ -f "visualisations/forward_diffusion.gif" ]; then
        echo "  ✓ PASSED"
        ((PASSED++))
    else
        echo "  ✗ FAILED (GIF not created)"
        ((FAILED++))
    fi
else
    echo "  ✗ FAILED (execution error or timeout)"
    ((FAILED++))
fi
echo ""

# Test 5: Create Animation (Reverse)
echo "Test 5: Create Animation (Reverse Process)"
echo "  Command: python create_animation.py --model_path models/DDPM_30epochs_cosine/ckpt.pt --mode reverse --frame_interval 10"
rm -f test_reverse.gif
if timeout 60 conda run -p $CONDA_ENV python create_animation.py --model_path models/DDPM_30epochs_cosine/ckpt.pt --mode reverse --frame_interval 10 --duration 100 > /dev/null 2>&1; then
    if [ -f "visualisations/reverse_diffusion.gif" ]; then
        echo "  ✓ PASSED"
        ((PASSED++))
    else
        echo "  ✗ FAILED (GIF not created)"
        ((FAILED++))
    fi
else
    echo "  ✗ FAILED (execution error or timeout)"
    ((FAILED++))
fi
echo ""

# Test 6: Unit Tests
echo "Test 6: Run Unit Tests"
echo "  Command: python test_conditional.py"
if conda run -p $CONDA_ENV python test_conditional.py > /dev/null 2>&1; then
    echo "  ✓ PASSED"
    ((PASSED++))
else
    echo "  ✗ FAILED (some tests failed)"
    ((FAILED++))
fi
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "⚠️  Some tests failed"
    exit 1
fi
