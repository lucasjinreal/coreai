@echo off
REM 检查setup.py配置
python3 setup.py check

REM 更新版本号（需提前安装bumpver）
bumpver update --patch

REM 清理旧构建文件
rd /s /q build
rd /s /q dist

REM 生成源码分发包
python3 setup.py sdist

REM 使用twine上传到PyPI
twine upload dist/*