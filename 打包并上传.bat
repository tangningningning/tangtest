@echo off
chcp 65001 >nul
echo ======================================
echo  正在为你生成服务器专用压缩包...
echo ======================================
echo.

REM 1. 删除可能存在的旧压缩包（避免混淆）
if exist adaptive_cache_mec.zip del adaptive_cache_mec.zip

REM 2. 使用Windows自带的tar命令打包（完美排除无用文件）
tar -a -cf adaptive_cache_mec.zip ^
    --exclude=".git" ^
    --exclude=".idea" ^
    --exclude="__pycache__" ^
    --exclude="*.pyc" ^
    --exclude="*.sif" ^
    --exclude="runs" ^
    --exclude="*.zip" ^
    --exclude=".vscode" ^
    ./*

echo.
echo ======================================
echo  ✅ 打包完成！文件位于当前文件夹：
echo     adaptive_cache_mec.zip
echo.
echo  📌 接下来请去网页端上传此文件。
echo ======================================
pause