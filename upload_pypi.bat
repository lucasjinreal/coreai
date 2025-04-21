@echo off
REM 检查setup.py配置
python setup.py check


REM 生成源码分发包
python setup.py sdist

REM 使用twine上传到PyPI
twine upload dist/*