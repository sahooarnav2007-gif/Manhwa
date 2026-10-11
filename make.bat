@echo off
set PYTHON=.venv\Scripts\python.exe
if not exist "%PYTHON%" set PYTHON=python

if "%1"=="setup" (
    "%PYTHON%" scripts/download_models.py && "%PYTHON%" scripts/make_folds.py
    exit /b %ERRORLEVEL%
)
if "%1"=="cache" (
    "%PYTHON%" -m src.eval.ablate --action cache --split development
    exit /b %ERRORLEVEL%
)
if "%1"=="weaklabels" (
    "%PYTHON%" -m src.eval.ablate --action weaklabels && "%PYTHON%" scripts/debug_render.py --num-pages 24
    exit /b %ERRORLEVEL%
)
if "%1"=="train" (
    "%PYTHON%" -m src.eval.ablate --action train
    exit /b %ERRORLEVEL%
)
if "%1"=="tune" (
    "%PYTHON%" -m src.eval.ablate --action tune
    exit /b %ERRORLEVEL%
)
if "%1"=="eval" (
    "%PYTHON%" -m src.eval.ablate --action eval
    exit /b %ERRORLEVEL%
)
if "%1"=="predict" (
    "%PYTHON%" -m src.eval.ablate --action predict
    exit /b %ERRORLEVEL%
)
if "%1"=="validate" (
    "%PYTHON%" -m src.pipeline.write_submission --submission submission.jsonl --sample dataset/sample_submission.jsonl
    exit /b %ERRORLEVEL%
)
if "%1"=="test" (
    "%PYTHON%" -m pytest -v
    exit /b %ERRORLEVEL%
)
echo Unknown target: %1
exit /b 1
