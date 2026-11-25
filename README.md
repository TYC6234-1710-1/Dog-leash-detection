# Dog leash detection

文明养犬场景下的犬只牵绳状态识别课程项目，使用 Python、Ultralytics 和 Streamlit。

## 当前进度

当前已完成中文图片推理界面，支持上传图片、选择模型、调整阈值，并显示中文检测框、类别数量和置信度。

目前使用通用预训练权重，可识别人、狗等通用目标。牵绳专用数据、训练权重、训练脚本和评测脚本仍待完成。dog_leashed、dog_unleashed 目前仅是中文显示映射，现有通用权重不会因此获得牵绳识别能力。

当前保留目标检测、实例分割、YOLO World 文本检测、YOLOE 文本分割、姿态估计这几种图片任务，视频与跟踪入口已移除。

## 课程任务

按每只犬输出位置和两类状态。

- dog_leashed 代表牵绳犬只，能确认绳连接犬身，另一端由人控制。
- dog_unleashed 代表未牵绳犬只，能确认未接绳，或另一端明确无人控制。

遮挡、出画或分辨率不足导致证据不足的图片另记原因，留作难例。主标注包含犬框和状态，保留完整原图。按拍摄场景或视频划分训练、验证、测试集，分别计算两类的查准率、查全率和 F1。

## 安装和启动

建议 Python 3.12。先安装 Git，依赖中的 CLIP 会通过 GitHub 下载。

```powershell
git clone https://github.com/TYC6234-1710-1/Dog-leash-detection.git
cd Dog-leash-detection
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

配置好环境后双击 start-original.bat，或在项目根目录执行：

```powershell
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) 'weights'
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

访问 http://127.0.0.1:8501 ，保持终端开启，退出时按 Ctrl+C。

macOS/Linux 可以创建并激活虚拟环境，安装 requirements.txt，再运行 python -m streamlit run app.py。

CPU 可以运行默认图片检测。如需 NVIDIA GPU，按本机驱动安装支持 CUDA 的 PyTorch，再检查：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 使用

1. 选择“目标检测”和“YOLO26-微型”。
2. 上传 JPG、JPEG、PNG、BMP、WEBP，或使用默认图片。
3. 调整置信度阈值，点击“开始识别”。
4. 查看检测框、类别统计和明细。

首次选用模型需要联网下载权重。YOLO World 和 YOLOE 还需下载文本编码器，首次加载较慢。权重在 weights/ 保存。

默认图片用于验证通用推理功能。上传图片目前不会自动保存为训练数据。更换图片、任务、模型、提示词或阈值后需要重新识别。

网页置信度由模型给出，滑块用于筛选预测框。查准率、查全率、F1 需使用带真实标注的测试集另行评估。

## 文件说明

- app.py 负责页面入口与侧栏。
- config.py 管理模型、路径、任务和默认阈值。
- image_service.py 处理图片读取、推理、中文标注及结果展示。
- model_loader.py 负责模型缓存、加载和设备选择。
- requirements.txt、pyproject.toml 声明依赖。
- start-original.bat 是 Windows 启动入口。
- images/office_4.jpg 是默认示例图，ATTRIBUTION.md 保留上游素材说明。
- weights/ 保存下载权重，其内容不纳入 Git。

## 后续工作

- 筛选牵绳数据，记录来源和许可。
- 整理标注规则并完成数据划分。
- 训练两类犬只状态检测模型。
- 生成独立测试集的逐类指标和错误样例。
- 接入自训权重，精简通用任务。

当前版本不包含已完成的牵绳训练或评测结果。

## 提交范围和运行说明

Git 排除虚拟环境、缓存、日志、环境变量文件、模型权重、数据集和训练输出。旧视频、教学笔记、截图及开题 PPT 不包含在首次代码提交中，本地仍保留。

依赖使用版本下限，尚未锁定完整环境。中文标注在 Windows 优先使用微软雅黑，在 Linux 尝试 Noto Sans CJK，其他环境可能需补充中文字体。

## 来源与许可

项目基于 [Aparsoft 的 yolo-streamlit-detection-tracking](https://github.com/aparsoft/yolo-streamlit-detection-tracking) 改造，保留 Apache-2.0 代码许可证。

主要改动为中文界面、中文类别与检测框、移除视频跟踪功能、本地启动脚本。默认图片和原素材说明来自上游，保留在 images/。上游未逐张给出示例图授权信息，素材权利以原作者说明为准。

Ultralytics、CLIP 及其权重与素材各自适用相应上游许可证。
