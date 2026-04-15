@echo off
REM Complete Phase 8: Interpretability & Visualization (Windows)
REM This script runs all steps to generate publication-ready figures

echo ========================================================================
echo PHASE 8: INTERPRETABILITY ^& VISUALIZATION
echo ========================================================================
echo.

REM Configuration
set CHECKPOINT=checkpoints\meta_learning\meta_learned_model.pt
set DATASET=davis
set EMBEDDINGS_DIR=embeddings
set OUTPUT_DIR=research_paper_visuals

REM Check if checkpoint exists
if not exist "%CHECKPOINT%" (
    echo Warning: Checkpoint not found at %CHECKPOINT%
    echo    Using default checkpoint path. Update if needed.
)

REM Step 1: Extract embeddings
echo ========================================================================
echo STEP 1: Extracting Embeddings
echo ========================================================================
echo.

python extract_embeddings.py ^
    --checkpoint "%CHECKPOINT%" ^
    --dataset "%DATASET%" ^
    --split test ^
    --output "%EMBEDDINGS_DIR%" ^
    --batch_size 256

if errorlevel 1 (
    echo Error: Embedding extraction failed
    pause
    exit /b 1
)

echo.
echo Embeddings extracted to %EMBEDDINGS_DIR%\
echo.

REM Step 2: Generate all visualizations
echo ========================================================================
echo STEP 2: Generating Visualizations
echo ========================================================================
echo.

python generate_visualizations.py ^
    --embeddings_dir "%EMBEDDINGS_DIR%" ^
    --checkpoint "%CHECKPOINT%" ^
    --dataset "%DATASET%" ^
    --output "%OUTPUT_DIR%" ^
    --num_attention_examples 5

if errorlevel 1 (
    echo Error: Visualization generation failed
    pause
    exit /b 1
)

echo.
echo Visualizations generated in %OUTPUT_DIR%\
echo.

REM Step 3: List generated files
echo ========================================================================
echo GENERATED FILES
echo ========================================================================
echo.

if exist "%OUTPUT_DIR%" (
    echo Files in %OUTPUT_DIR%\:
    dir /b "%OUTPUT_DIR%\*.png" 2>nul
    if errorlevel 1 echo   (No PNG files found)
) else (
    echo Warning: Output directory not found: %OUTPUT_DIR%
)

echo.
echo ========================================================================
echo PHASE 8 COMPLETE!
echo ========================================================================
echo.
echo Generated visualizations:
echo   - Drug/protein UMAP and t-SNE plots
echo   - Attention heatmaps (5 examples)
echo   - Multi-head attention comparisons
echo   - Uncertainty calibration plots
echo.
echo These are ready for your paper!
echo.
echo Next steps:
echo   1. Review generated figures in %OUTPUT_DIR%\
echo   2. Select best examples for paper
echo   3. Add figure captions
echo   4. Include in manuscript
echo.

pause
