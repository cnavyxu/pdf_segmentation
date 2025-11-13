# 银行对账单智能解析系统 - 代码逻辑文档

## 项目概述

本项目是一个基于Python Flask框架的银行对账单智能解析系统，主要功能包括多家银行对账单的PDF解析、OCR识别、表格提取、数据结构化等服务。

### 技术栈
- **Web框架**: Flask + Flask-CORS
- **PDF处理**: PyMuPDF (fitz), pdfplumber, PyPDF2
- **图像处理**: OpenCV (cv2), PIL, pypdfium2
- **OCR识别**: 自定义OCR模块
- **表格识别**: 自定义深度学习模型（ONNX格式）
- **数据处理**: NumPy, Pandas

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Flask Web 服务                          │
│                   (donghang_deploy.py)                       │
├─────────────────────────────────────────────────────────────┤
│  API端点:                                                     │
│  - /huaqiao, /meixi, /huaqi (各银行专用接口)                │
│  - /cut_pdf, /cut_pdf_mutipage (PDF裁剪)                    │
│  - /render, /render_file (渲染服务)                         │
│  - /classify_pdf (PDF分类)                                   │
│  - /huifeng_merge, /mogeng_merge (智能路由)                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    核心处理模块                              │
├─────────────────────────────────────────────────────────────┤
│  ├─ pdf_process.py           (PDF解析和OCR处理)             │
│  ├─ pdf_process_pymu.py      (PyMuPDF OCR处理)             │
│  ├─ extract_*_utils.py       (各银行数据提取模块)           │
│  ├─ utils/cut_pdf.py         (PDF智能裁剪)                  │
│  ├─ utils/formal_exact.py    (结构化数据提取)               │
│  └─ utils/classify_utils.py  (文档分类)                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    AI模型层                                  │
├─────────────────────────────────────────────────────────────┤
│  ├─ table_predict_main.py    (表格识别主模块)               │
│  │   ├─ TABLE_LOCATION       (表格定位模型)                 │
│  │   ├─ TABLE_STRUCT_LINED   (有线表格结构识别)             │
│  │   └─ TABLE_STRUCT_BORDERLESS (无线表格结构识别)          │
│  └─ model/                   (ONNX模型文件)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 主要模块详解

### 1. donghang_deploy.py - Flask Web服务主文件

#### 1.1 全局配置
```python
# Flask应用初始化
app = Flask(__name__)
CORS(app, supports_credentials=True)  # 启用跨域资源共享
app.config["JSON_AS_ASCII"] = False   # 支持中文JSON响应
port = 3019                            # 服务监听端口

# 表格识别模型初始化
table_model = table_predict_main.TABLE(
    gpu_id=0,                          # GPU设备ID（0表示使用CPU）
    location_model="model/best.onnx",  # 表格定位模型
    struct_model="model/table_line.onnx"  # 表格结构识别模型
)
```

#### 1.2 核心函数

##### pdf2images_fit(b_str_image, ratio=1.7)
**功能**: 将PDF字节流转换为图像列表

**处理流程**:
1. 使用PyMuPDF打开PDF字节流
2. 遍历每一页，按指定缩放比例渲染
3. 将渲染结果转换为OpenCV图像格式
4. 返回图像列表

**参数说明**:
- `b_str_image`: PDF文件的二进制字节流
- `ratio`: 缩放比例（默认1.7），控制输出图像分辨率

**返回值**: 图像列表（每个元素为numpy数组）

##### create_id()
**功能**: 生成基于时间戳的唯一MD5哈希ID

**用途**: 用于生成文件名、会话ID等唯一标识符

##### bs64toimg(str_image)
**功能**: Base64字符串转换为OpenCV图像对象

**处理流程**:
1. Base64字符串解码为字节流
2. 使用OpenCV解码为图像数组

##### ronghui_process(res, config)
**功能**: 融汇银行对账单数据提取处理

**处理流程**:
1. 获取坐标偏移和锚点位置
2. 根据配置提取字段值
3. 验证提取结果完整性
4. 执行主提取流程
5. 获取字段位置映射
6. 封装最终结果

**参数说明**:
- `res`: OCR识别结果
- `config`: 提取配置（包含字段定义和规则）

#### 1.3 API端点详解

##### /huaqiao - 华侨银行对账单处理
**请求方式**: POST/GET

**请求参数**:
- `file`: PDF文件（multipart/form-data）
- `save_dir`: 结果保存目录

**处理流程**:
1. 接收上传的PDF文件
2. 执行OCR识别获取文本和坐标信息
3. 调用华侨银行专用提取模块 `extract_huaqiao.extract_all()`
4. 将PDF转换为高清图像（ratio=4.0）
5. 遍历每笔交易记录，渲染单独的明细图片
6. 将明细图片转换为PDF格式并保存
7. 使用 `ronghui_process` 提取表头信息
8. 返回JSON格式的结构化数据

**返回数据结构**:
```json
{
  "code": 200,
  "message": "ok",
  "data": {
    "item": {},           // 表头信息字段
    "ocr_data": "",       // OCR原始数据
    "head_h": 0,          // 表头高度
    "form": [],           // 交易明细列表
    "shape_list": []      // 页面尺寸列表
  }
}
```

##### /meixi - 美西银行对账单处理
**功能**: 类似华侨银行接口，使用美西银行专用提取逻辑

**特殊处理**:
- 使用 `extract_meixi_utils.extract_all()` 提取数据
- 使用 `extract_meixi_utils.process_biaotou()` 处理表头

##### /cut_pdf - PDF智能裁剪（单页）
**请求方式**: POST/GET

**请求参数**:
- `file`: PDF文件
- `upcut_str`: 上边界匹配字符串（正则表达式）
- `downcut_str`: 下边界匹配字符串（正则表达式）
- `up_offset`: 上边界偏移量（像素）
- `down_offset`: 下边界偏移量（像素）
- `upcut_filter`: 上边界过滤字符串
- `downcut_filter`: 下边界过滤字符串
- `ocr_url`: OCR服务URL

**处理流程**:
1. 解析裁剪配置参数
2. 对PDF执行OCR识别
3. 根据正则表达式匹配上下边界
4. 考虑过滤条件排除误匹配
5. 计算裁剪区域
6. 执行PDF裁剪
7. 返回Base64编码的裁剪后PDF

**应用场景**: 去除对账单中的页眉页脚、广告等无关内容

##### /cut_pdf_mutipage - PDF智能裁剪（多页）
**功能**: 支持跨页裁剪，可将一个PDF拆分成多个子PDF

**返回值**: Base64编码的PDF列表

##### /sanjin_minxi - 三金银行明细对账单
**特殊处理**:
- 使用 `pdf_process_pymu.get_ocr_pymu()` 进行OCR（PyMuPDF方式）
- 适用于特殊格式的银行对账单

##### /sanjin_duizhang - 三金银行对账单
**特殊处理**:
- 使用 `pypdfium2` 进行高质量PDF渲染（scale=7）
- 处理高分辨率扫描件

##### /huaqi, /deyizhi, /bmo, /taiwan, /zhaoshang - 各银行专用接口
**共同特点**:
- 统一的处理流程
- 各自独立的提取逻辑模块
- 差异在于字段配置和解析规则

##### /huifeng_ch, /huifeng_en - 汇丰银行（中英文版本）
**智能处理**:
- 根据语言自动调用不同的解析器
- `/huifeng_ch` 处理中文版本
- `/huifeng_en` 处理英文版本

##### /mogeng_ch, /mogeng_en - 摩根银行（中英文版本）
**特殊处理**:
- 自动检测 "客户参考" 或 "Customer Ref" 关键字
- 计算裁剪偏移量 `op_cut`
- 根据语言调用不同的表头处理函数

##### /huaqi_wuliu - 华旗物流对账单
**智能分类**:
```python
# 检测对账单类型
class_ind = 0
for one_line_text in ocr_data[0]["text"]:
    if "账户名" in one_line_text:
        class_ind = 1  # 类型2
        break

# 根据类型选择处理器
if class_ind == 0:
    res_form, iih = extract_wuliu_huaqi.extract_all()
else:
    res_form, iih = extract_wuliu_huaqi2.extract_all()
```

##### /huifeng_yinni - 汇丰印尼对账单（表格识别）
**高级功能**: 使用深度学习表格识别模型

**处理流程**:
1. 对PDF执行OCR识别
2. 将PDF转换为图像并调整尺寸（min_len=1600）
3. 调用表格识别模型 `table_model.table_predict()`
4. 解析HTML表格结构
5. 定位每个单元格坐标
6. 提取表头区域
7. 遍历每一行数据
8. 渲染单独的明细图片（包含表头+当前行）
9. 保存为PDF格式

**表格处理关键代码**:
```python
# 表格识别
contour_analyser_list = table_model.table_predict(img, ocr_data[p])

for contour_analyser_i in contour_analyser_list:
    # 解析HTML表格
    df = pd.read_html(StringIO(contour_analyser_i.html))[0]
    
    # 清理数据
    df = df.applymap(lambda x: x.replace("\n", " "))
    
    # 构建位置索引映射
    pos2ind_map = {}
    for i, pot in enumerate(contour_analyser_i.cell_table_coord):
        pos2ind_map[tuple(list(pot))] = i
    
    # 提取表头区域
    top_region = [cell_hull_coord[ind] for col in range(columns)]
    biaotou_span = min_max_h_span(top_region)
    
    # 遍历数据行
    for row in range(1, rows):
        # 检测终止标记
        if cell_text in ["PHTotal:", "SGTotal:", ...]:
            break
        
        # 渲染明细图片
        img_merge = concatenate([img_header, img_row])
```

##### /is_pdf_parse - PDF可解析性检测
**功能**: 检测PDF是否可以通过文本方式解析（非扫描件）

**检测逻辑**:
```python
# 执行OCR
ocr_data, shape_list = pdf_process.get_ocr(f_obj)

# 统计第一页有效文本行数
has_ind_num = 0
for line_text in ocr_data[0]["text"]:
    if len(line_text) > 0:
        has_ind_num += 1

# 判断可解析性
if has_ind_num >= 3:
    is_pdf_parse = 1  # 可解析
else:
    is_pdf_parse = 0  # 需要OCR识别
```

##### /render, /render_file - 渲染服务
**功能**: 根据坐标信息在PDF图像上绘制标注框

**输入参数**:
- `file`: PDF文件（Base64或路径）
- `form`: 标注数据
- `thickness`: 线条粗细
- `page`: 页码
- `line_ind`: 行索引
- `shape_list`: 页面尺寸
- `head_h`: 表头高度

**返回值**: Base64编码的标注后图像

##### /classify_pdf - PDF文档分类
**功能**: 根据配置的关键字规则对PDF进行分类

**配置示例**:
```json
{
  "page": 0,
  "data": {
    "en": {
      "da": ["Description Customer Ref"],
      "no": ["客户参考 银行参考"]
    },
    "ch": {
      "da": ["客户参考 银行参考"],
      "no": ["Description Customer Ref"]
    }
  }
}
```

**分类逻辑**:
```python
ocr_data, shape_list = pdf_process.get_ocr(f_obj)
class_name = classify_utils.classify_run(ocr_data, config_json)
# class_name 为 "en" 或 "ch"
```

##### /mogeng_merge, /huifeng_merge - 智能路由接口
**功能**: 自动检测文档语言，路由到对应的处理接口

**处理流程**:
1. 调用 `/classify_pdf` 接口分类
2. 根据分类结果 (`ch` 或 `en`)
3. 内部转发到对应的处理接口
4. 返回处理结果

**示例（摩根银行）**:
```python
# 第一步：分类
response = requests.post("/classify_pdf", data={
    "config": '{"page":0,"data":{"en":{"da":["Description Customer Ref"]}}}'
})
class_result = response.json()["data"]

# 第二步：路由
if class_result == "ch":
    response = requests.post("/mogeng_ch", files=files)
else:
    response = requests.post("/mogeng_en", files=files)
```

##### /huifeng_cut - 汇丰对账单智能拆分
**功能**: 根据关键字 "AdviceType通知种类" 将多账户对账单拆分

**拆分逻辑**:
```python
# 检测关键字所在页码
biyou_id_list = []
for i, ocr_data_i in enumerate(ocr_data):
    text_all = "".join(ocr_data_i["raw_text"])
    if "AdviceType通知种类" in text_all:
        biyou_id_list.append(i)

# 生成拆分范围
res_span = []
for i in range(len(biyou_id_list)):
    if i + 1 < len(biyou_id_list):
        res_span.append((biyou_id_list[i], biyou_id_list[i + 1]))
    else:
        res_span.append((biyou_id_list[i], len(ocr_data)))

# 使用PyPDF2拆分
pdf_reader = PdfReader(f_obj)
for span in res_span:
    pdf_writer = PdfWriter()
    for page_num in range(span[0], span[1]):
        pdf_writer.add_page(pdf_reader.pages[page_num])
    # 输出为Base64
```

#### 1.4 辅助函数

##### fix_min_len_resize(img, min_l, max_len=2500)
**功能**: 图像等比例缩放（保证最短边不小于min_l）

**计算公式**:
```python
ratio = min(min_l / min(h, w), max_len / max(h, w))
new_h, new_w = int(ratio * h), int(ratio * w)
```

##### min_max_h_span(box_list)
**功能**: 计算多个矩形框的垂直跨度（最小y和最大y）

**用途**: 用于确定表格行的高度范围

---

### 2. pdf_process.py - PDF解析和OCR处理模块

#### 2.1 核心数据结构

**OCR数据结构**:
```python
{
    'span': [],      # 字符中心点坐标列表 [(x1,y1), (x2,y2), ...]
    'text': [],      # 文本行列表 ["line1", "line2", ...]
    'ind': [],       # 字符索引列表
    'blob': [],      # 文本块列表（包含box和text）
    'raw_text': []   # 原始文本（未处理）
}
```

**Blob数据结构**:
```python
{
    'box': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  # 四点坐标
    'text': "text content",                        # 文本内容
    'span': [(x1,y1), (x2,y2), ...]               # 字符中心点
}
```

#### 2.2 坐标转换函数

##### two_to_four_convert(points, ratio)
**功能**: 将2点坐标（左上、右下）转换为4点坐标（逆时针）

**转换逻辑**:
```
输入: [x0, y0, x1, y1]
输出: [[x0,y0], [x1,y0], [x1,y1], [x0,y1]]
同时应用缩放比例 ratio
```

**用途**: 统一坐标格式，便于绘制和计算

#### 2.3 文本聚合函数

##### w_cluster_has_blank(cluster_obj, r_shape, ratio, crop_box)
**功能**: 将PDF字符聚合成文本行（保留空格）

**处理流程**:
1. 遍历每一行的字符对象
2. 按x坐标排序
3. 调用 `move_overlap()` 去除重叠字符
4. 根据字符间距判断是否属于同一个文本块
   - 间距 ≤ 2: 合并到当前块
   - 间距 > 2: 创建新文本块
5. 计算文本块的四点坐标
6. 记录字符中心点坐标

**关键参数**:
- `cluster_obj`: pdfplumber聚合的字符对象
- `r_shape`: 页面尺寸 (width, height)
- `ratio`: 缩放比例
- `crop_box`: 裁剪区域（可选）

##### w_cluster(cluster_obj, r_shape, ratio, page_box)
**功能**: 将PDF字符聚合成文本行（不保留空格）

**差异**:
- 间距阈值为 8（更宽松）
- 过滤超出页面边界的字符

##### move_overlap(ts_line)
**功能**: 去除同一行中的重叠字符（PDF加粗效果）

**检测逻辑**:
```python
# 计算重叠度（IOU）
iou = (min(x1_end, x2_end) - max(x1_start, x2_start)) / width

if iou > 0.7:
    # 重叠度大于70%，认为是重复字符，跳过
    continue
```

#### 2.4 图像提取函数

##### ReadPdfImage(filepath)
**功能**: 从PDF中提取并重建图像（处理旋转、翻转、透明度）

**复杂处理流程**:

1. **获取页面画布尺寸**:
```python
canvos_point1 = page.CropBox.top_left
canvos_point2 = page.CropBox.bottom_right
canvos = np.ones([h, w, 3]) * 255  # 白色画布
```

2. **计算全局缩放比例**:
```python
def get_r_ratio(image_info):
    # 找出图像的最大缩放比例
    for img_info in image_info:
        tmp_h, tmp_w = bbox的宽高
        g_h, g_w = 图像实际宽高
        ratio = max(g_w/tmp_w, g_h/tmp_h)
    return max_ratio
```

3. **处理图像变换**:
```python
# 水平翻转
if transform[0] < 0:
    img = np.flip(img, axis=1)

# 垂直翻转
if transform[3] < 0:
    img = np.flip(img, axis=0)

# 颜色空间转换
if colorspace == 0:  # 反色
    img = 255 - img
```

4. **处理重名图像**:
```python
# PDF中可能有同名图像对象，只处理唯一的
img_name_num_map = defaultdict(int)
for im_info in image_list:
    img_name_num_map[im_info[7]] += 1

if img_name_num_map[item_name] >= 2:
    continue  # 跳过重复图像
```

5. **图像拼接到画布**:
```python
# 计算有效区域
max_h = min(canvos.shape[0] - point1[1], img.shape[0])
max_w = min(canvos.shape[1] - point1[0], img.shape[1])

# 拼接
canvos[y1:y1+max_h, x1:x1+max_w, :] = img[0:max_h, 0:max_w, :]
```

6. **处理页面旋转**:
```python
rotate_angle = get_rotate_angle(page.derotation_matrix)
r_canvos = np.rot90(canvos, rotate_angle // 90)
```

##### get_rotate_angle(der_matrix)
**功能**: 从变换矩阵计算旋转角度

**计算方法**:
```python
# 枚举可能的旋转角度
rot_list = [0, 90, 180, 270]

# 计算矩阵差异
for rot in rot_list:
    rot_matrix = fitz.Matrix(rot)
    dist = sum(abs(der_matrix[i] - rot_matrix[i]))
    
# 返回差异最小的角度
```

#### 2.5 主要接口函数

##### get_ocr(pdf_path)
**功能**: 从PDF文件提取文本和坐标（pdfplumber方式）

**处理流程**:
1. 使用pdfplumber打开PDF
2. 遍历每一页
3. 获取页面尺寸和裁剪框
4. 计算缩放比例（保证最短边1600px）
5. 如果存在字符对象，执行聚合
6. 调用 `w_cluster_has_blank()` 聚合文本行
7. 调用 `wrap_sorted_line_pdf()` 封装数据结构

**返回值**: (ocr_data, shape_list)

**应用场景**: 处理可提取文本的PDF（非扫描件）

##### get_ocr_not_has_blank(pdf_path)
**功能**: 类似get_ocr，但不保留空格

**差异**:
- 使用 `w_cluster()` 替代 `w_cluster_has_blank()`
- 不进行缩放（ratio=1.0）
- 适用于精确定位场景

##### wrap_sorted_line_pdf(sorted_line)
**功能**: 将聚合的文本行封装为标准数据结构

**处理内容**:
1. 合并同一行的所有文本块
2. 替换中文标点为英文标点（统一格式）
3. 记录字符索引位置
4. 过滤空行

**标点替换**:
```python
text = text.replace('（', '(')
           .replace('）', ')')
           .replace('，', ',')
```

##### get_pdf_ratio(page_shape)
**功能**: 计算PDF页面的标准化缩放比例

**目标**: 
- 最短边不小于1600px
- 最长边不超过2500px

**公式**:
```python
ratio = min(1600/min(w,h), 2500/max(w,h))
```

---

### 3. utils/cut_pdf.py - PDF智能裁剪模块

#### 3.1 核心功能

该模块实现了基于关键字匹配的PDF智能裁剪功能，主要用于：
- 去除对账单页眉页脚
- 提取特定区域内容
- 拆分多账户对账单

#### 3.2 裁剪配置结构

```python
split_config = {
    'upcut_str': '.*',          # 上边界正则表达式
    'downcut_str': '.*',        # 下边界正则表达式
    'up_offset': 10,            # 上边界偏移（像素）
    'down_offset': 10,          # 下边界偏移（像素）
    'upcut_filter': '.*',       # 上边界过滤正则
    'downcut_filter': '.*'      # 下边界过滤正则
}
```

#### 3.3 核心函数详解

##### get_span_list_single(pattern_str, img, ocr_data, page, up_offset, down_offset, split_config)
**功能**: 单关键字裁剪（只指定上边界，自动延伸到页面底部）

**处理流程**:
```python
1. 编译正则表达式
pattern = re.compile(pattern_str)

2. 遍历OCR文本行，匹配关键字
for text in ocr_data['text']:
    if pattern.search(text):
        # 检查过滤条件
        if upcut_filter.search(text):
            continue  # 跳过
        
        # 记录匹配行的y坐标
        up_h = ocr_data['blob'][i][0]['box'][0][1]
        split_list.append(up_h)

3. 图像二值化检测实际内容区域
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dd = np.where(img_gray < 100)  # 找到所有深色像素
have_pix_max = np.max(dd[0])   # 最大y坐标

4. 生成裁剪范围
for i, split_point in enumerate(split_list):
    if i == last:
        span = (split_point - up_offset, have_pix_max + down_offset)
    else:
        span = (split_point - up_offset, split_list[i+1] - up_offset)
    span_list.append(span)
```

**应用场景**: 处理 "从第X页开始" 类型的裁剪

##### get_span_list_double(page, up_cut_str, down_cut_str, ocr_data, up_offset, down_offset, split_config)
**功能**: 双关键字裁剪（指定上下边界）

**匹配逻辑**:
```python
1. 找出所有上边界匹配位置
up_tmp_list = [i for i, text in enumerate(texts) 
               if up_pattern.match(text) and not upcut_filter.match(text)]

2. 找出所有下边界匹配位置
down_tmp_list = [i for i, text in enumerate(texts)
                 if down_pattern.match(text) and not downcut_filter.match(text)]

3. 配对上下边界
for k, up_pos in enumerate(up_tmp_list):
    next_up_pos = up_tmp_list[k+1] if k+1 < len else 10000
    
    # 找到up_pos之后、next_up_pos之前的最后一个down_pos
    last_down = None
    for down_pos in down_tmp_list:
        if up_pos < down_pos < next_up_pos:
            last_down = down_pos
    
    if last_down:
        tmp_merge_list.append((up_pos, last_down))

4. 计算裁剪坐标
for (up_ind, down_ind) in tmp_merge_list:
    up_y = ocr_data['blob'][up_ind][0]['box'][0][1] - up_offset
    down_y = ocr_data['blob'][down_ind][0]['box'][2][1] + down_offset
    span_list.append((up_y, down_y))
```

**应用场景**: 提取 "开始标记" 到 "结束标记" 之间的内容

##### get_span_list_merge(split_config, img, ocr_data, page)
**功能**: 智能选择单关键字或双关键字裁剪

**决策逻辑**:
```python
if len(downcut_str) > 0:
    # 双关键字模式
    span_list = get_span_list_double(...)
else:
    # 单关键字模式
    span_list = get_span_list_single(...)
```

##### wrap_ocr_res_by_ratio(ocr_data, ratio)
**功能**: 根据缩放比例调整OCR坐标

**处理对象**:
1. `blob['box']`: 文本框四点坐标
2. `span`: 字符中心点坐标

**转换公式**:
```python
new_box = [[x * ratio, y * ratio] for [x, y] in box]
new_span = [int(x * ratio), int(y * ratio) for [x, y] in span]
```

**用途**: 
- 图像缩放后同步调整坐标
- OCR服务返回的坐标适配到原始尺寸

##### crop_and_save_pdf(input_path, output_path, split_config, ocr_data, img_list, http_url, up_cut_str, down_cut_str)
**功能**: 执行PDF裁剪并保存

**详细处理流程**:

```python
# 1. 打开PDF文档
pdf_document = fitz.Document(input_path)
new_pdf_document = fitz.Document()

# 2. 遍历每一页
for page_number in range(pdf_document.pageCount):
    page = pdf_document[page_number]
    
    # 3. 检测是否可以直接解析
    text_all = "".join(ocr_data[page_number]["text"])
    is_ok = True
    
    if up_cut_str != "" and not up_pattern.search(text_all):
        is_ok = False
    if down_cut_str != "" and not down_pattern.search(text_all):
        is_ok = False
    
    # 4. 如果不可解析，调用HTTP OCR服务
    if not is_ok or len(ocr_data[page_number]["text"]) == 0:
        # 将图像编码为Base64
        img_save = cv2.imencode('.jpg', img)[1]
        b64_str = base64.b64encode(img_save.tostring())
        
        # 请求OCR服务
        response = requests.post(http_url, json={
            "is_pdf_parse": 0,
            "file": b64_str
        })
        
        # 解析返回的OCR数据
        ocr_data[page_number] = wrap_ocr_res_by_ratio(
            response.json()["data"], 
            ratio
        )
    
    # 5. 计算裁剪范围
    span_list = get_span_list_merge(split_config, img, ocr_data[page_number], page)
    
    # 6. 执行裁剪
    for (h1, h2) in span_list:
        if 0 <= h1 <= h2 <= page_height:
            # 定义裁剪区域
            clip_rect = fitz.Rect(0, h1, page_width, h2)
            
            # 创建新页面
            new_page = new_pdf_document.new_page(
                width=page_width, 
                height=h2 - h1
            )
            
            # 复制裁剪区域到新页面
            new_page.show_pdf_page(
                fitz.Rect(0, 0, page_width, h2 - h1),
                pdf_document,
                page_number
            )

# 7. 保存PDF
new_pdf_bytesio = io.BytesIO()
new_pdf_document.save(new_pdf_bytesio)
new_pdf_bytes = new_pdf_bytesio.getvalue()

# 8. 清理资源
new_pdf_document.close()
pdf_document.close()

return new_pdf_bytes
```

**关键点**:
- 使用 `show_pdf_page()` 复制页面内容（保留矢量信息）
- 支持OCR降级（PDF无法解析时调用OCR服务）
- 返回字节流（不直接写文件）

##### crop_and_save_pdf_opt(...)
**功能**: 优化版PDF裁剪（修复内存泄漏）

**改进点**:
1. **使用 `page_count` 替代 `pageCount`** （新版API）
2. **使用 `tobytes()` 替代 `tostring()`** （过时方法）
3. **添加页面数量验证**:
```python
page_added = False

for span in span_list:
    # 裁剪页面
    ...
    page_added = True

if not page_added or new_pdf_document.page_count == 0:
    raise ValueError("No valid pages after cropping")
```

4. **使用临时文档避免修改原始PDF**:
```python
temp_doc = fitz.Document()
temp_page = temp_doc.new_page(...)
temp_page.show_pdf_page(...)

# 插入到目标文档
new_pdf_document.insert_pdf(temp_doc, ...)

# 释放临时文档
temp_doc.close()
del temp_doc
gc.collect()
```

5. **添加资源释放保障**:
```python
finally:
    if new_pdf_document:
        new_pdf_document.close()
    if pdf_document:
        pdf_document.close()
    if new_pdf_bytesio:
        new_pdf_bytesio.close()
    gc.collect()
```

##### get_mutipage_cut_span(up_cut_str, up_filter, down_cut_str, down_filter, ocr_data)
**功能**: 计算跨页裁剪范围（页码级别）

**处理模式**:

**模式1: 双关键字（is_double=True）**
```python
# 找到所有上边界页码
up_match_list = [page_num for page_num, page_data in enumerate(ocr_data)
                 if up_pattern.match(any_line) and not up_filter.match(any_line)]

# 找到所有下边界页码
down_match_list = [page_num ...]

# 配对
for up_page in up_match_list:
    for down_page in down_match_list:
        if down_page >= up_page:
            span_list.append((up_page, down_page))
            break
```

**模式2: 单关键字（is_double=False）**
```python
# 找到所有关键字页码
up_match_list = [page_num ...]

# 生成范围
for i, up_page in enumerate(up_match_list):
    if i == last:
        span_list.append((up_page, total_pages - 1))
    else:
        span_list.append((up_page, up_match_list[i+1]))
```

**返回值**: [(start_page, end_page), ...]

##### crop_and_save_pdf_multipage(...)
**功能**: 执行跨页裁剪，返回多个PDF（Base64列表）

**处理流程**:
```python
1. 计算跨页裁剪范围
span_list = get_mutipage_cut_span(...)

2. 对每个范围拆分PDF
for (start_page, end_page) in span_list:
    pdf_writer = PdfWriter()
    
    for page_num in range(start_page, end_page + 1):
        pdf_writer.add_page(pdf_reader.pages[page_num])
    
    # 保存到BytesIO
    output = BytesIO()
    pdf_writer.write(output)
    pdf_bytes = output.getvalue()
    
    # Base64编码
    base64_list.append(base64.b64encode(pdf_bytes).decode())

3. 返回Base64列表
return base64_list
```

**应用场景**: 将多账户对账单拆分为独立文件

---

### 4. utils/table_predict/table_predict_main.py - 表格识别模块

#### 4.1 模块架构

```
TABLE (主类)
├─ table_location         (表格定位模型)
├─ table_struct_lined     (有线表格结构识别)
└─ table_struct_borderless (无线表格结构识别)
```

#### 4.2 TABLE类

##### __init__(gpu_id, location_model_path, struct_lined_model_path)
**功能**: 初始化表格识别模型

**加载模型**:
```python
# 表格定位模型（YOLO v7）
self.table_location = TABLE_LOCATION(
    gpu_id=gpu_id,
    model_path=location_model_path  # best.onnx
)

# 有线表格结构识别模型
self.table_struct_lined = TABLE_STRUCT_LINED(
    gpu_id=gpu_id,
    model_path=struct_lined_model_path  # table_line.onnx
)
```

##### merge(table_points, cut_points_list, dist_)
**功能**: 将单元格坐标从截取图坐标系转换为原图坐标系

**转换逻辑**:
```python
# 1. 计算表格左上角坐标
min_w = min([pt[0] for pt in table_points])
min_h = min([pt[1] for pt in table_points])

# 2. 计算截取时的边框宽度
cut_iw = max(min_w - dist_, 0)
cut_ih = max(min_h - dist_, 0)

# 3. 转换每个单元格坐标
cell_bbox_list = []
for cut_bbox in cut_points_list:
    cell_bbox = [[pt[0] + cut_iw, pt[1] + cut_ih] for pt in cut_bbox]
    cell_bbox_list.append(cell_bbox)

return cell_bbox_list
```

**参数说明**:
- `table_points`: 表格在原图中的顶点坐标
- `cut_points_list`: 单元格在截取表格图中的坐标
- `dist_`: 截取表格时额外截取的边框宽度
  - 有线表格: 64 像素
  - 无线表格: 0 像素

##### table_predict(img, res_ocr_i=None)
**功能**: 表格识别主流程

**完整处理流程**:

```python
# 1. 图像预处理
img_pdf, _ = fix_min_len_resize(img, 1012)   # 用于模型推理
img, _ = fix_min_len_resize(img, 1600)       # 用于显示
ratio = img_pdf.shape[0] / img.shape[0]

# 2. 表格定位
location_result = self.table_location.predict(img)
# 返回: [{'label': 'LINED'/'BORDERLESS', 'bbox': [x1,y1,x2,y2]}, ...]

# 3. 提取表格区域
shapes = []
for table_result in location_result:
    table_label = table_result['label']
    table_bbox = table_result['bbox']
    table_points = [[table_bbox[0], table_bbox[1]], 
                    [table_bbox[2], table_bbox[3]]]
    shapes.append({
        'table_points': table_points,
        'table_label': table_label
    })

img_cut_list = generate_function.get_img_cut(img, shapes)

# 4. 遍历每个表格
contour_analyzer_list = []
for img_cut_info in img_cut_list:
    table_label = img_cut_info['table_label']
    table_points = img_cut_info['table_points']
    img_cut = img_cut_info['img_cut']
    
    if table_label == BORDERLESS_LABEL:
        # 4.1 无线表格处理
        dist_ = DIST_DICT[table_label]  # 0
        
        # 结构识别
        structure_str_list, cut_bbox_list = \
            self.table_struct_borderless.predict(img_cut)
        
        # 转换为四点坐标
        cut_points_list = [f_rect2poly(bbox) for bbox in cut_bbox_list]
        
        # 生成HTML
        html_content = ''.join(structure_str_list)
        
        # 坐标转换
        cell_points_list = self.merge(table_points, cut_points_list, dist_)
        
        # 对齐单元格坐标
        cell_list, has_merged_cells, column_num, row_num = \
            f_aligning(html_content, cell_points_list)
        
        # 后处理
        contour_analyzer = f_borderless_postprocessing(
            res_ocr_i, cell_list
        )
        
    elif table_label == LINED_LABEL:
        # 4.2 有线表格处理
        dist_ = DIST_DICT[table_label]  # 64
        
        # 结构识别
        cut_bbox_list = self.table_struct_lined.predict(img_cut, ratio)
        
        # 转换坐标
        cut_points_list = [f_rect2poly(bbox) for bbox in cut_bbox_list]
        cell_points_list = self.merge(table_points, cut_points_list, dist_)
        
        # 对齐有线表格
        cell_list, row_num, column_num, box_dict_origin, box_dict_new, performance = \
            f_bbox2cell(cell_points_list)
        
        # 后处理
        contour_analyzer = f_lined_postprocessing(res_ocr_i, cell_list)
        contour_analyzer.box_dict_origin = box_dict_origin
        contour_analyzer.box_dict_new = box_dict_new
    
    # 5. 封装结果
    contour_analyzer.row_num = row_num
    contour_analyzer.column_num = column_num
    contour_analyzer.cell_list = cell_list
    
    # 生成HTML
    html_list = f_cell2html(cell_list, need_text=True)
    contour_analyzer.html_list = html_list
    contour_analyzer.html = '<table border="1">' + ''.join(html_list) + '</table>'
    
    # 保存其他信息
    contour_analyzer.res_ocr_i = {'blob': copy.deepcopy(res_ocr_i['blob'])}
    contour_analyzer.img_origin = img
    contour_analyzer.table_label = table_label
    contour_analyzer.table_points = table_points
    contour_analyzer.performance = performance
    contour_analyzer.img_cut = img_cut
    contour_analyzer.cut_points_list = cut_points_list
    
    contour_analyzer_list.append(contour_analyzer)

# 6. 返回所有表格的识别结果
return contour_analyzer_list
```

#### 4.3 输出数据结构

**ContourAnalyzer对象**:
```python
{
    'row_num': 5,                  # 表格行数
    'column_num': 3,               # 表格列数
    'cell_list': [...],            # 单元格列表
    'html': '<table>...</table>',  # HTML格式表格
    'html_list': [...],            # HTML标签列表
    'cell_table_coord': [...],     # 单元格表格坐标 (col, row)
    'cell_hull_coord': [...],      # 单元格图像坐标 [[x,y], ...]
    'res_ocr_i': {...},            # OCR数据
    'img_origin': img,             # 原始图像
    'table_label': 'LINED',        # 表格类型
    'table_points': [...],         # 表格位置
    'performance': 'good',         # 识别质量
    'img_cut': img,                # 截取的表格图像
    'cut_points_list': [...]       # 单元格坐标列表
}
```

**cell_list结构**:
```python
[
    {
        'row': 0,                  # 行号
        'col': 0,                  # 列号
        'row_span': 1,             # 行合并数
        'col_span': 1,             # 列合并数
        'bbox': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  # 四点坐标
        'text': 'cell content'     # 单元格文本
    },
    ...
]
```

#### 4.4 表格识别流程图

```
原始图像
    ↓
图像缩放 (fix_min_len_resize)
    ↓
表格定位 (TABLE_LOCATION)
    ↓
提取表格区域 (get_img_cut)
    ↓
    ├─ 无线表格 (BORDERLESS)
    │   ├─ 结构识别 (TABLE_STRUCT_BORDERLESS)
    │   ├─ 生成HTML
    │   ├─ 单元格对齐 (f_aligning)
    │   └─ 后处理 (f_borderless_postprocessing)
    │
    └─ 有线表格 (LINED)
        ├─ 结构识别 (TABLE_STRUCT_LINED)
        ├─ 单元格对齐 (f_bbox2cell)
        └─ 后处理 (f_lined_postprocessing)
    ↓
生成HTML (f_cell2html)
    ↓
返回 ContourAnalyzer 对象
```

---

## 数据流示意图

### 完整业务流程（以华侨银行为例）

```
1. 客户端上传PDF
    ↓
2. Flask接收 (/huaqiao)
    ↓
3. PDF → OCR识别
   pdf_process.get_ocr(file)
   返回: ocr_data, shape_list
    ↓
4. 数据提取
   extract_huaqiao.extract_all(ocr_data, shape_list)
   返回: res_form, iih
    ↓
5. PDF → 高清图像
   pdf2images_fit(pdf_bytes, ratio=4.0)
    ↓
6. 渲染明细图片
   for page in res_form:
       for item in page['data']:
           img = render_one_item(imgs, res_form, shape_list, iih, p, q)
           pdf_bytes = img2pdf.convert(img)
           save(pdf_bytes)
           item['img_path'] = path
    ↓
7. 提取表头信息
   itemss = ronghui_process(ocr_data[0], config)
    ↓
8. 封装返回结果
   {
     "code": 200,
     "data": {
       "item": itemss,
       "form": res_form,
       "ocr_data": json.dumps(ocr_data),
       "head_h": iih,
       "shape_list": shape_list
     }
   }
    ↓
9. 返回JSON响应
```

### PDF裁剪流程

```
1. 上传PDF + 裁剪配置
    ↓
2. OCR识别
   get_ocr(pdf_path)
    ↓
3. 匹配关键字
   for text in ocr_data['text']:
       if pattern.match(text):
           记录y坐标
    ↓
4. 计算裁剪范围
   span_list = [(y1, y2), (y3, y4), ...]
    ↓
5. 执行裁剪
   for (h1, h2) in span_list:
       clip_rect = fitz.Rect(0, h1, w, h2)
       new_page.show_pdf_page(clip_rect)
    ↓
6. 保存新PDF
   new_pdf_bytes = BytesIO()
    ↓
7. Base64编码返回
   return base64.b64encode(new_pdf_bytes)
```

### 表格识别流程

```
1. 图像输入
    ↓
2. 表格定位
   YOLO检测表格区域
    ↓
3. 截取表格区域
   img_cut = img[y1:y2, x1:x2]
    ↓
4. 表格结构识别
   ├─ 无线表格: SLANet模型
   └─ 有线表格: 线条检测算法
    ↓
5. 单元格定位
   cut_bbox_list = [(x1,y1,x2,y2), ...]
    ↓
6. 坐标对齐
   对齐到表格行列结构
    ↓
7. OCR文本填充
   for cell in cells:
       cell['text'] = match_ocr_text(cell['bbox'], ocr_data)
    ↓
8. 生成HTML
   html = f_cell2html(cell_list)
    ↓
9. 返回结果
   ContourAnalyzer对象
```

---

## 关键技术点

### 1. 坐标系统

项目中使用多个坐标系统，需要正确转换：

**PDF坐标系**（pdfplumber）:
- 原点：左上角
- X轴：向右
- Y轴：向下
- 单位：点（1/72英寸）

**图像坐标系**（OpenCV）:
- 原点：左上角
- X轴：向右
- Y轴：向下
- 单位：像素

**PyMuPDF坐标系**（fitz）:
- 原点：左上角
- X轴：向右
- Y轴：向下

**坐标转换**:
```python
# PDF → 图像坐标
img_x = pdf_x * ratio
img_y = (pdf_page_height - pdf_y) * ratio  # Y轴需要翻转

# 四点坐标格式
[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]  # 逆时针
```

### 2. 文本提取策略

**策略1: pdfplumber提取（文本PDF）**
- 优点：快速、准确、保留格式
- 缺点：仅适用于可提取文本的PDF
- 使用场景：银行电子对账单

**策略2: PyMuPDF提取（混合PDF）**
- 优点：支持图像和文本混合
- 缺点：需要额外处理
- 使用场景：扫描件+文本的PDF

**策略3: HTTP OCR服务（扫描件）**
- 优点：适用于所有PDF
- 缺点：速度慢、需要网络
- 使用场景：纯图像PDF、降级方案

**自动选择逻辑**:
```python
# 尝试提取文本
ocr_data, shape_list = pdf_process.get_ocr(file)

# 检测是否可解析
text_count = sum(1 for text in ocr_data[0]['text'] if len(text) > 0)

if text_count >= 3:
    # 可解析，使用提取的文本
    use_extracted_text()
else:
    # 不可解析，调用OCR服务
    response = requests.post(ocr_url, json={"file": base64_image})
    ocr_data = response.json()['data']
```

### 3. 表格识别技术

**有线表格识别**:
1. **线条检测**: 使用形态学操作检测水平和垂直线
2. **交点计算**: 计算线条交点作为单元格顶点
3. **单元格聚类**: 根据交点构建单元格
4. **行列对齐**: 对齐到规则的行列结构

**无线表格识别**:
1. **深度学习模型**: SLANet模型识别表格结构
2. **HTML生成**: 模型输出HTML标签序列
3. **坐标预测**: 预测每个单元格的边界框
4. **单元格对齐**: 根据HTML结构对齐坐标

### 4. 内存管理

**问题**: 处理大量PDF时容易内存泄漏

**解决方案**:
```python
# 1. 及时关闭文档对象
pdf_document.close()
del pdf_document

# 2. 使用临时文档
temp_doc = fitz.Document()
# ... 操作 ...
temp_doc.close()
del temp_doc

# 3. 定期垃圾回收
import gc
gc.collect()

# 4. 使用上下文管理器
with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
    # ... 操作 ...
    # 自动清理

# 5. 清理PyMuPDF缓存
fitz.TOOLS.store_shrink(100)
```

### 5. 正则表达式最佳实践

**避免误匹配**:
```python
# 问题：过于宽泛
pattern = re.compile(".*账单.*")  # 可能匹配太多

# 改进：更精确
pattern = re.compile("^账单日期[:：].*$")  # 以"账单日期"开头

# 使用过滤器排除误匹配
upcut_filter = re.compile("备注|说明")
if pattern.match(text) and not upcut_filter.match(text):
    # 正确的匹配
```

**处理空格**:
```python
# 移除所有空格后匹配
text_clean = text.replace(" ", "")
if pattern.search(text_clean):
    # 匹配成功
```

### 6. PDF裁剪精度控制

**偏移量设计**:
```python
# 上边界偏移（通常为负值，向上扩展）
up_offset = -10  # 向上扩展10像素，避免裁掉关键字

# 下边界偏移（通常为正值，向下扩展）
down_offset = 10  # 向下扩展10像素，保留完整内容
```

**坐标验证**:
```python
# 确保坐标在有效范围内
def wrap_h(page_h, d):
    if d <= 0:
        return 0
    if d >= page_h:
        return int(page_h)
    return int(d)
```

---

## 性能优化建议

### 1. PDF处理优化

**批量处理**:
```python
# 避免重复打开PDF
pdf = pdfplumber.open(file)
for page in pdf.pages:
    # 处理每一页
pdf.close()

# 而不是
for i in range(page_count):
    pdf = pdfplumber.open(file)  # 重复打开
    page = pdf.pages[i]
    pdf.close()
```

**图像缓存**:
```python
# 一次性转换所有页
imgs = pdf2images_fit(pdf_bytes, ratio=4.0)

# 后续重复使用
for p in range(len(res_form)):
    img = imgs[p]  # 直接使用缓存
```

### 2. OCR优化

**分辨率控制**:
```python
# 根据场景选择合适的分辨率
pdf2images_fit(pdf_bytes, ratio=1.7)  # 一般场景
pdf2images_fit(pdf_bytes, ratio=4.0)  # 高精度场景
pdf2images_fit(pdf_bytes, ratio=1.0)  # 性能优先
```

**并行处理**:
```python
# 多页PDF并行OCR
from multiprocessing import Pool

def ocr_page(page):
    return pdf_process.get_ocr(page)

with Pool(4) as pool:
    results = pool.map(ocr_page, pages)
```

### 3. 模型推理优化

**批量推理**:
```python
# 收集所有表格
img_cut_list = [...]

# 批量推理（如果模型支持）
results = model.predict_batch(img_cut_list)
```

**模型量化**:
```python
# 使用INT8量化模型
table_model = TABLE(
    location_model="best_int8.onnx",  # 量化模型
    struct_model="table_line_int8.onnx"
)
```

---

## 常见问题及解决方案

### 1. 坐标不准确

**问题**: 提取的文本框坐标与实际位置不符

**原因**:
- PDF页面有旋转
- PDF页面有裁剪框
- 缩放比例不正确

**解决方案**:
```python
# 检查页面属性
if page.cropbox is None:
    r_w, r_h = page.bbox[2], page.bbox[3]
    crop_box = page.bbox
else:
    r_w, r_h = page.cropbox[2] - page.cropbox[0], page.cropbox[3] - page.cropbox[1]
    crop_box = page.cropbox

# 处理旋转
rotate_angle = get_rotate_angle(page.derotation_matrix)
img = np.rot90(img, rotate_angle // 90)
```

### 2. 文本识别不完整

**问题**: OCR漏识别部分文本

**原因**:
- 图像分辨率太低
- 字符间距过大被拆分
- 特殊字符无法识别

**解决方案**:
```python
# 提高分辨率
imgs = pdf2images_fit(pdf_bytes, ratio=4.0)  # 增加到4.0

# 调整聚合阈值
if obj['x0'] - tmp_list[-1][2] <= 8:  # 增加阈值
    # 合并到同一个文本块
```

### 3. 表格识别失败

**问题**: 表格结构识别错误

**原因**:
- 表格边界不清晰
- 单元格合并复杂
- 表格旋转或倾斜

**解决方案**:
```python
# 检查识别质量
if contour_analyzer.performance == 'bad':
    # 使用备用方案
    fallback_process()

# 手动调整
if has_merged_cells:
    # 特殊处理合并单元格
```

### 4. 内存泄漏

**问题**: 长时间运行后内存占用增加

**原因**:
- PyMuPDF对象未释放
- 图像缓存未清理
- 循环引用

**解决方案**:
```python
# 使用crop_and_save_pdf_opt替代crop_and_save_pdf
# 定期清理
gc.collect()
fitz.TOOLS.store_shrink(100)

# 使用弱引用
import weakref
cache = weakref.WeakValueDictionary()
```

### 5. PDF裁剪后为空

**问题**: 裁剪后的PDF没有内容

**原因**:
- 关键字匹配失败
- 坐标计算错误
- span_list为空

**解决方案**:
```python
# 添加验证
if not span_list:
    logging.warning(f"No valid spans found for page {page_num}")
    # 使用整页
    span_list = [(0, page_height)]

# 验证坐标
if not (0 <= h1 < h2 <= page_height):
    logging.error(f"Invalid coordinates: h1={h1}, h2={h2}")
    continue
```

---

## 扩展开发指南

### 1. 添加新银行支持

**步骤**:

1. **创建提取模块** `extract_newbank_utils.py`:
```python
def extract_all(ocr_data, shape_list):
    """提取表格数据"""
    res_form = []
    # 实现提取逻辑
    return res_form, head_height

def process_biaotou(ocr_data):
    """提取表头信息"""
    itemss = {}
    # 实现提取逻辑
    return itemss
```

2. **添加API端点** `donghang_deploy.py`:
```python
@app.route("/newbank", methods=["POST", "GET"])
def post_Data_Cetifas_newbank():
    f_obj = request.files.get("file")
    save_dir = request.form.get("save_dir")
    
    # OCR识别
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)
    
    # 数据提取
    res_form, iih = extract_newbank_utils.extract_all(ocr_data, shape_list)
    
    # ... 渲染和保存 ...
    
    return jsonify(res)
```

3. **配置路由（如需中英文支持）**:
```python
@app.route("/newbank_merge", methods=["POST", "GET"])
def newbank_merge():
    # 分类
    class_result = classify_pdf(file, config)
    
    # 路由
    if class_result == "ch":
        return forward_to("/newbank_ch")
    else:
        return forward_to("/newbank_en")
```

### 2. 自定义裁剪规则

**场景**: 需要更复杂的裁剪逻辑

**实现**:

```python
def custom_get_span_list(ocr_data, custom_config):
    """
    自定义裁剪规则
    custom_config: {
        'rules': [
            {'type': 'keyword', 'value': '开始'},
            {'type': 'regex', 'value': r'\d{4}-\d{2}-\d{2}'},
            {'type': 'position', 'value': (100, 200)}
        ]
    }
    """
    span_list = []
    
    for rule in custom_config['rules']:
        if rule['type'] == 'keyword':
            # 关键字匹配
            for i, text in enumerate(ocr_data['text']):
                if rule['value'] in text:
                    y = ocr_data['blob'][i][0]['box'][0][1]
                    span_list.append(y)
        
        elif rule['type'] == 'regex':
            # 正则匹配
            pattern = re.compile(rule['value'])
            for i, text in enumerate(ocr_data['text']):
                if pattern.search(text):
                    y = ocr_data['blob'][i][0]['box'][0][1]
                    span_list.append(y)
        
        elif rule['type'] == 'position':
            # 固定位置
            span_list.append(rule['value'][0])
    
    return span_list
```

### 3. 添加新表格类型支持

**步骤**:

1. **训练新模型**:
```python
# 训练YOLO模型识别新表格类型
# 训练结构识别模型
```

2. **更新配置**:
```python
# utils/table_predict/utils/local_settings.py
NEW_TABLE_LABEL = "NEW_TYPE"
DIST_DICT[NEW_TABLE_LABEL] = 32  # 边框宽度
```

3. **添加处理逻辑**:
```python
# table_predict_main.py
elif table_label == NEW_TABLE_LABEL:
    # 新表格处理逻辑
    contour_analyzer = new_table_process(img_cut)
```

---

## 部署指南

### 1. 环境配置

**依赖安装**:
```bash
pip install flask flask-cors
pip install PyMuPDF pdfplumber PyPDF2 pypdfium2
pip install opencv-python numpy pandas
pip install img2pdf
pip install onnxruntime  # CPU版本
# pip install onnxruntime-gpu  # GPU版本
```

**模型文件**:
```
model/
├── best.onnx           # 表格定位模型
└── table_line.onnx     # 表格结构模型
```

### 2. 启动服务

**开发环境**:
```bash
python donghang_deploy.py
# 服务监听: http://0.0.0.0:3019
```

**生产环境（使用Gunicorn）**:
```bash
gunicorn -w 4 -b 0.0.0.0:3019 donghang_deploy:app \
    --timeout 300 \
    --max-requests 1000 \
    --max-requests-jitter 50
```

**Docker部署**:
```dockerfile
FROM python:3.8

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 3019

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:3019", "donghang_deploy:app"]
```

### 3. 配置参数

**donghang_deploy.py**:
```python
# 端口号
port = 3019

# 模型路径
table_model = table_predict_main.TABLE(
    gpu_id=0,  # GPU ID（-1为CPU）
    location_model_path="model/best.onnx",
    struct_lined_model_path="model/table_line.onnx"
)
```

**性能调优**:
```python
# Flask配置
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 最大16MB

# 超时设置
import socket
socket.setdefaulttimeout(300)
```

---

## 测试建议

### 1. 单元测试

**测试PDF处理**:
```python
import unittest

class TestPdfProcess(unittest.TestCase):
    def test_get_ocr(self):
        ocr_data, shape_list = pdf_process.get_ocr("test.pdf")
        self.assertTrue(len(ocr_data) > 0)
        self.assertTrue(len(shape_list) > 0)
    
    def test_coordinate_conversion(self):
        points = [0, 0, 100, 100]
        result = two_to_four_convert(points, 1.0)
        expected = [[0,0], [100,0], [100,100], [0,100]]
        np.testing.assert_array_equal(result, expected)
```

**测试裁剪功能**:
```python
def test_cut_pdf(self):
    split_config = {
        'upcut_str': '账单开始',
        'downcut_str': '账单结束',
        'up_offset': -10,
        'down_offset': 10
    }
    
    result = crop_and_save_pdf(
        input_path="test.pdf",
        split_config=split_config,
        ...
    )
    
    self.assertIsNotNone(result)
    self.assertTrue(len(result) > 0)
```

### 2. 集成测试

**测试API端点**:
```python
import requests

def test_huaqiao_api():
    url = "http://localhost:3019/huaqiao"
    
    with open("test.pdf", "rb") as f:
        files = {"file": f}
        data = {"save_dir": "/tmp/test"}
        response = requests.post(url, files=files, data=data)
    
    assert response.status_code == 200
    result = response.json()
    assert result['code'] == 200
    assert 'data' in result
```

### 3. 性能测试

**测试处理速度**:
```python
import time

def test_performance():
    start = time.time()
    
    for i in range(100):
        ocr_data, shape_list = pdf_process.get_ocr("test.pdf")
    
    elapsed = time.time() - start
    avg_time = elapsed / 100
    
    print(f"Average processing time: {avg_time:.2f}s")
    assert avg_time < 1.0  # 要求平均1秒内完成
```

---

## 维护和监控

### 1. 日志记录

**添加日志**:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.route("/huaqiao", methods=["POST"])
def post_Data_Cetifas_huaqiao():
    logger.info(f"Received request from {request.remote_addr}")
    try:
        # 处理逻辑
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise
```

### 2. 错误处理

**统一错误响应**:
```python
@app.errorhandler(Exception)
def handle_error(e):
    logger.error(f"Unhandled exception: {e}")
    return jsonify({
        'code': 500,
        'message': str(e),
        'data': None
    }), 500
```

### 3. 性能监控

**添加性能指标**:
```python
import time

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        logger.info(f"{func.__name__} took {elapsed:.2f}s")
        
        # 发送到监控系统
        # send_to_prometheus(func.__name__, elapsed)
        
        return result
    return wrapper

@monitor_performance
def process_pdf(file):
    # 处理逻辑
    pass
```

---

## 总结

本系统是一个功能完善的银行对账单智能解析平台，具有以下特点：

**优势**:
1. 支持多家银行对账单格式
2. 灵活的PDF裁剪和处理能力
3. 先进的表格识别技术
4. 完善的坐标转换和数据提取
5. 良好的扩展性

**适用场景**:
- 银行对账单自动化处理
- 财务数据批量提取
- 表格图像识别
- PDF文档智能裁剪

**持续改进方向**:
1. 优化模型精度和速度
2. 增加更多银行支持
3. 改进内存管理
4. 增强错误处理
5. 完善监控和日志

---

**文档版本**: 1.0  
**最后更新**: 2024年  
**维护团队**: 算法团队
