"""图片上传、模型推理与中文结果展示。"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd
import PIL.Image
import PIL.ImageOps
from PIL import ImageDraw, ImageFont
import streamlit as st

import config
from model_loader import SHARED_MODEL_LOCK, get_model_for_task

# COCO 的 80 类名称只用于展示，不改变模型类别编号和推理结果。
_ENGLISH = 'person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush'.split(',')
_CHINESE = '人,自行车,汽车,摩托车,飞机,公交车,火车,卡车,船,交通信号灯,消防栓,停车标志,停车计时器,长椅,鸟,猫,狗,马,羊,牛,大象,熊,斑马,长颈鹿,背包,雨伞,手提包,领带,行李箱,飞盘,滑雪板,单板滑雪板,运动球,风筝,棒球棒,棒球手套,滑板,冲浪板,网球拍,瓶子,葡萄酒杯,杯子,叉子,刀,勺子,碗,香蕉,苹果,三明治,橙子,西兰花,胡萝卜,热狗,披萨,甜甜圈,蛋糕,椅子,沙发,盆栽,床,餐桌,马桶,电视,笔记本电脑,鼠标,遥控器,键盘,手机,微波炉,烤箱,烤面包机,水槽,冰箱,书,时钟,花瓶,剪刀,玩具熊,吹风机,牙刷'.split(',')
_NAMES = dict(zip(_ENGLISH, _CHINESE))
_NAMES.update({'table': '桌子', 'phone': '手机', 'dog_leashed': '牵绳犬只', 'dog_unleashed': '未牵绳犬只'})


def _name(value: str) -> str:
    return _NAMES.get(value, value)


@st.cache_resource(show_spinner=False)
def _label_font(size: int):
    for path in (Path('C:/Windows/Fonts/msyh.ttc'), Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def _annotated_image(result):
    # 分割掩膜和关键点仍由模型绘制，框上的类别改用支持中文的字体。
    image = PIL.Image.fromarray(result.plot(labels=False)[:, :, ::-1])
    draw = ImageDraw.Draw(image)
    font = _label_font(max(16, min(image.size) // 35))
    if result.boxes is not None:
        for box in result.boxes:
            x, y, _, _ = box.xyxy[0].cpu().tolist()
            label = f'{_name(result.names[int(box.cls.item())])} {box.conf.item():.2f}'
            bounds = draw.textbbox((0, 0), label, font=font)
            width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
            x = max(0, min(int(x), image.width - width - 8))
            y = max(0, min(int(y) - height - 10, image.height - height - 8))
            draw.rectangle((x, y, x + width + 8, y + height + 8), fill=(45, 55, 65))
            draw.text((x + 4, y + 4 - bounds[1]), label, font=font, fill='white')
    return image


def render(task: str, confidence: float, selected_model: str | None = None) -> None:
    st.header(f'图片处理 · {task}')
    # Streamlit 上传控件的内置英文提示通过局部样式替换。
    st.html('''<style>
    [data-testid="stFileUploaderDropzoneInstructions"] > * {display:none;}
    [data-testid="stFileUploaderDropzoneInstructions"]::after {content:"支持 JPG、PNG、BMP、WEBP，单张不超过 200 MB";font-size:12px;}
    [data-testid="stFileUploaderDropzone"] button {font-size:0;}
    [data-testid="stFileUploaderDropzone"] button * {display:none;}
    [data-testid="stFileUploaderDropzone"] button::after {content:"选择图片";font-size:14px;}
    [data-testid="stToolbar"] {display:none;}
    </style>''')
    classes = None
    if task in (config.TASK_WORLD, config.TASK_YOLOE):
        classes = _class_input(task)
        if not classes:
            return
    model = get_model_for_task(task, classes, model_name=selected_model)
    if model is None:
        return
    uploaded = st.file_uploader('上传图片', type=['jpg', 'jpeg', 'png', 'bmp', 'webp'], key='img_upload')
    if uploaded:
        image_bytes = uploaded.getvalue()
        caption = '原始图片'
    elif config.DEFAULT_IMAGE.exists():
        image_bytes = config.DEFAULT_IMAGE.read_bytes()
        caption = '示例图片，可在上方上传自己的图片'
    else:
        st.info('请先上传一张图片。')
        return
    try:
        image = _open_image(image_bytes)
    except (OSError, ValueError):
        st.error('无法读取图片，请检查文件是否完整，或换一张图片。')
        return
    run_key = hashlib.md5(image_bytes + repr((task, selected_model, classes, confidence)).encode()).hexdigest()
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption=caption, width='stretch')
    with col2:
        if st.button('开始识别', type='primary', width='stretch'):
            with st.spinner('正在识别，请稍候……'):
                st.session_state['_image_result'] = (run_key, _predict(model, image, confidence))
        saved = st.session_state.get('_image_result')
        if saved and saved[0] == run_key:
            result = saved[1]
            st.image(_annotated_image(result), caption=f'{task}结果', width='stretch')
            _display_results(result, task)
        else:
            st.info('点击“开始识别”后，结果会显示在这里。')


def _class_input(task: str) -> list[str] | None:
    if task == config.TASK_YOLOE:
        st.caption('输入英文类别名，用逗号分隔。例如 person 表示人，dog 表示狗。此模式同时输出检测框和分割区域。')
        default = config.DEFAULT_YOLOE_CLASSES
    else:
        st.caption('输入英文类别名或简短描述，用逗号分隔。例如 dog 表示狗，red car 表示红色汽车。未检出目标时可适当降低阈值。')
        default = config.DEFAULT_WORLD_CLASSES
    text = st.text_area('检测类别', value=default, help='模型使用英文提示词，请保留英文类别输入。')
    classes = [c.strip() for c in text.replace('，', ',').split(',') if c.strip()]
    if classes:
        st.info('当前检测类别：' + '、'.join(_name(c) for c in classes))
    else:
        st.warning('请至少输入一个检测类别。')
    return classes or None


def _open_image(data: bytes) -> PIL.Image.Image:
    return PIL.ImageOps.exif_transpose(PIL.Image.open(io.BytesIO(data))).convert('RGB')


def _predict(model, image: PIL.Image.Image, confidence: float):
    with SHARED_MODEL_LOCK:
        return model.predict(image, conf=confidence, verbose=False)[0]


def _display_results(result, task: str) -> None:
    with st.expander('识别结果', expanded=True):
        if result.boxes is not None and len(result.boxes):
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            summary = {}
            for cls_id, conf in zip(classes, confs):
                name = _name(result.names[int(cls_id)])
                entry = summary.setdefault(name, {'count': 0, 'total_conf': 0.0})
                entry['count'] += 1
                entry['total_conf'] += float(conf)
            cols = st.columns(min(len(summary), 4))
            for idx, (name, data) in enumerate(summary.items()):
                with cols[idx % len(cols)]:
                    st.metric(label=name, value=data['count'])
                    st.caption(f"平均置信度：{data['total_conf'] / data['count']:.0%}")
            rows = [{'类别': _name(result.names[int(c)]), '置信度': f'{cf:.2%}'} for c, cf in zip(classes, confs)]
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        if task == config.TASK_POSE:
            kp = getattr(result, 'keypoints', None)
            if kp is not None and len(kp):
                st.success(f'检测到 {len(kp)} 个人体姿态。')
            else:
                st.info('未检测到人体姿态，可适当降低置信度阈值。')
        if task in (config.TASK_SEGMENT, config.TASK_YOLOE):
            masks = getattr(result, 'masks', None)
            if masks is not None and len(masks):
                st.success(f'已分割 {len(masks)} 个目标。')
        if result.boxes is None or len(result.boxes) == 0:
            st.info('未检测到目标，可适当降低置信度阈值。')
