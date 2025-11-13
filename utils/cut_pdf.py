import pdfplumber
import pdf_process_pymu
import pdf_process
import fitz  # 导入PyMuPDF库用于PDF处理
import cv2  # 导入OpenCV计算机视觉库
import numpy as np  # 导入NumPy库用于数值计算和数组操作
import re  # 导入正则表达式模块
import io
import requests  # 导入HTTP请求库
import json  # 导入JSON数据处理库
import base64  # 导入base64库用于Base64编码解码操作
import copy


def pdf2images_fit(b_str_image):  # 定义函数
    pdf_doc = fitz.open(stream=b_str_image, filetype="bytes")  # 打开文件或资源
    img_list = []  # 变量赋值
    for i in range(pdf_doc.pageCount):  # 循环遍历
        page = pdf_doc[i]  # 变量赋值
        zoom_x = zoom_y = 1.7  # 变量赋值
        mat = fitz.Matrix(zoom_x, zoom_y)  # 变量赋值
        pix = page.getPixmap(matrix=mat)  # 变量赋值
        tt = pix.tobytes()  # 转换为字节串
        img_decode = cv2.imdecode(np.frombuffer(tt, np.uint8), cv2.IMREAD_COLOR)  # 将字节流解码为图像
        img_list.append(img_decode)  # 添加元素到列表
    return img_list  # 返回结果


def wrap_h(page_h, d):  # 定义函数
    if d <= 0:  # 条件判断
        return 0  # 返回结果
    if d >= page_h:  # 条件判断
        return int(page_h)  # 返回结果
    return int(d)  # 返回结果


def generate_span(split_list, have_pix_max, page_h, up_offset, down_offset):  # 定义函数
    res = []  # 变量赋值
    for i, plit_d in enumerate(split_list):  # 循环遍历
        if i == len(split_list) - 1:  # 条件判断
            res.append(  # 添加元素到列表
                (
                    wrap_h(page_h, split_list[i] - up_offset),
                    wrap_h(page_h, have_pix_max + down_offset),
                )
            )
        else:  # 否则执行
            res.append(  # 添加元素到列表
                (
                    wrap_h(page_h, split_list[i] - up_offset),
                    wrap_h(page_h, split_list[i + 1]) - up_offset,
                )
            )
    return res  # 返回结果


def get_span_list_single(  # 定义函数
    pattern_str, img, ocr_data, page, up_offset, down_offset, split_config
):
    pattern = re.compile(pattern_str)  # 变量赋值
    split_list = []  # 变量赋值
    for i, text in enumerate(ocr_data["text"]):  # 循环遍历
        dt = pattern.search(text.replace(" ", ""))  # 变量赋值
        if not dt is None:  # 条件判断

            fu_pattern = re.compile(split_config["upcut_filter"])  # 变量赋值
            if not fu_pattern.search(text) is None:  # 条件判断
                continue
            up_h = ocr_data["blob"][i][0]["box"][0][1]  # 取 匹配字符串box 上部
            split_list.append(up_h)  # 添加元素到列表
    page_h, page_w = page.rect[3], page.rect[2]  # 变量赋值
    img = cv2.resize(img, (int(page_w), int(page_h)))  # 变量赋值
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 变量赋值
    dd = np.where(img_gray < 100)  # 变量赋值
    have_pix_max = np.max(dd[0])  # 变量赋值
    span_list = generate_span(split_list, have_pix_max, page_h, up_offset, down_offset)  # 变量赋值
    return span_list  # 返回结果


def get_span_list_double(  # 定义函数
    page, up_cut_str, down_cut_str, ocr_data, up_offset, down_offset, split_config
):
    up_pattern = re.compile(up_cut_str)  # 变量赋值
    down_pattern = re.compile(down_cut_str)  # 变量赋值
    up_tmp_list = []  # 变量赋值
    down_tmp_list = []  # 变量赋值
    for i, text in enumerate(ocr_data["text"]):  # 循环遍历
        if not up_pattern.search(text.replace(" ", "")) is None:  # 条件判断
            if split_config["upcut_filter"].replace(" ", "") == "":  # 条件判断
                pass
            else:  # 否则执行
                fu_pattern = re.compile(split_config["upcut_filter"])  # 变量赋值
                if not fu_pattern.search(text) is None:  # 条件判断
                    continue
            up_tmp_list.append(i)  # 添加元素到列表
        if not down_pattern.search(text.replace(" ", "")) is None:  # 条件判断
            if split_config["downcut_filter"].replace(" ", "") == "":  # 条件判断
                pass
            else:  # 否则执行
                fd_pattern = re.compile(split_config["downcut_filter"])  # 变量赋值
                if not fd_pattern.search(text) is None:  # 条件判断
                    continue
            down_tmp_list.append(i)  # 添加元素到列表
    tmp_merge_list = []  # 变量赋值
    for k, p in enumerate(up_tmp_list):  # 循环遍历
        if k == len(up_tmp_list) - 1:  # 条件判断
            next_ind = 10000  # 变量赋值
        else:  # 否则执行
            next_ind = up_tmp_list[k + 1]  # 变量赋值
        last_q = None  # 变量赋值
        for q_k, q in enumerate(down_tmp_list):  # 循环遍历
            if q >= next_ind:  # 条件判断
                break
            if q > p:  # 条件判断
                last_q = q  # 变量赋值
        if not last_q is None:  # 条件判断
            tmp_merge_list.append((p, last_q))  # 添加元素到列表

    span_list = []  # 变量赋值
    page_h, page_w = page.rect[3], page.rect[2]  # 变量赋值
    for tt_span in tmp_merge_list:  # 循环遍历
        uup = tt_span[0]  # 变量赋值
        ddp = tt_span[1]  # 变量赋值
        uup_point = wrap_h(page_h, ocr_data["blob"][uup][0]["box"][0][1] - up_offset)  # 变量赋值

        ddp_point = wrap_h(page_h, ocr_data["blob"][ddp][0]["box"][2][1] + down_offset)  # 变量赋值
        span_list.append((uup_point, ddp_point))  # 添加元素到列表
    return span_list  # 返回结果


def get_span_list_merge(split_config, img, ocr_data, page):  # 定义函数
    upcut_str = split_config["upcut_str"]  # 变量赋值
    downcut_str = split_config["downcut_str"]  # 变量赋值
    up_offset = split_config["up_offset"]  # 变量赋值
    down_offset = split_config["down_offset"]  # 变量赋值
    if len(downcut_str.replace(" ", "")) > 0:  # 条件判断
        span_list = get_span_list_double(  # 变量赋值
            page, upcut_str, downcut_str, ocr_data, up_offset, down_offset, split_config
        )
    else:  # 否则执行
        span_list = get_span_list_single(  # 变量赋值
            upcut_str, img, ocr_data, page, up_offset, down_offset, split_config
        )
    return span_list  # 返回结果


def fix_min_len_resize(x_shape, min_l, max_len=2500):  # 定义函数
    h, w = x_shape  # 变量赋值
    ratio = float(min_l) / min(h, w)  # 变量赋值
    ratio = min(max_len / max(h, w), ratio)  # 变量赋值
    new_h, new_w = int(ratio * h), int(ratio * w)  # 变量赋值
    res_ratio = (float(w) / new_w, float(h) / new_h)  # 变量赋值
    return res_ratio  # 返回结果


def wrap_ocr_res_by_ratio(ocr_data, ratio):  # 定义函数
    res = copy.deepcopy(ocr_data)  # 变量赋值
    for one_line_blob in res["blob"]:  # 循环遍历
        for blob in one_line_blob:  # 循环遍历
            box = blob["box"]  # 变量赋值
            box = np.asarray(box)  # 变量赋值
            box = box * ratio  # 变量赋值
            blob["box"] = box.tolist()  # 转换为Python列表

    for one_line_span in res["span"]:  # 循环遍历
        for k, span_k in enumerate(one_line_span):  # 循环遍历
            new_span = [int(span_k[0] * ratio), int(span_k[1] * ratio)]  # 变量赋值
            one_line_span[k] = new_span  # 变量赋值
    return res  # 返回结果


def crop_and_save_pdf(  # 定义函数
    input_path,
    output_path,
    split_config,
    ocr_data,
    img_list,
    http_url,
    up_cut_str,
    down_cut_str,
):
    # 打开 PDF 文件
    pdf_document = fitz.Document(input_path)  # 变量赋值
    new_pdf_document = fitz.Document()  # 变量赋值
    up_cut_pattern = re.compile(up_cut_str)  # 变量赋值
    down_cut_pattern = re.compile(down_cut_str)  # 变量赋值
    # 获取指定页数的页面对象
    ppk = []  # 变量赋值
    for page_number in range(pdf_document.pageCount):  # 循环遍历
        page = pdf_document[page_number]  # 页数从0开始

        # 获取页面的高度
        page_height = page.rect.height  # 变量赋值
        page_width = page.rect.width  # 变量赋值
        img = img_list[page_number]  # 变量赋值
        ocr_data_i = ocr_data[page_number]  # 变量赋值
        # 如果指定的高度范围有效，则进行裁剪
        is_ok = True  # 变量赋值
        text_all = "".join(ocr_data_i["text"])  # 变量赋值
        if not up_cut_str == "":  # 条件判断
            dt = up_cut_pattern.search(text_all)  # 变量赋值
            if dt is None:  # 条件判断
                is_ok = False  # 变量赋值
        if not down_cut_str == "":  # 条件判断
            dt = down_cut_pattern.search(text_all)  # 变量赋值
            if dt is None:  # 条件判断
                is_ok = False  # 变量赋值
        if len(ocr_data_i["text"]) == 0 or is_ok == False:  # 条件判断
            img_save = cv2.imencode(".jpg", img)[1]  # 将图像编码为指定格式
            data_encode = np.array(img_save)  # 创建NumPy数组
            str_encode = data_encode.tostring()  # 转换为字节字符串
            b64_str_encode = base64.b64encode(str_encode).decode("utf-8")  # 对字节数据进行Base64编码
            ratio_w_h = fix_min_len_resize((page_height, page_width), 1600)  # 变量赋值
            headers = {"Content-Type": "application/json"}  # 变量赋值
            payload = json.dumps({"is_pdf_parse": 0, "file": b64_str_encode})  # 将对象序列化为JSON字符串
            response = requests.request("POST", http_url, headers=headers, data=payload)  # 变量赋值
            http_json_res = json.loads(response.text)  # 将JSON字符串解析为对象
            ocr_data_i = wrap_ocr_res_by_ratio(http_json_res["data"], ratio_w_h[0])  # 变量赋值
        span_list = get_span_list_merge(split_config, img, ocr_data_i, page)  # 变量赋值
        ppk.append(span_list)  # 添加元素到列表
        for span in span_list:  # 循环遍历
            h1 = span[0]  # 变量赋值
            h2 = span[1]  # 变量赋值

            if 0 <= h1 <= h2 <= page_height:  # 条件判断
                # 构建一个矩形区域对象，定义裁剪区域
                clip_rect = fitz.Rect(0, h1, page.rect.width, h2)  # 变量赋值
                # 裁剪页面
                page.set_cropbox(clip_rect)
                new_page = new_pdf_document.new_page(  # 变量赋值
                    width=page.rect.width, height=h2 - h1  # 变量赋值
                )
                # 创建新的 PDF 文档
                # new_page.delete_text(clip_rect)
                # 将裁剪后的页面添加到新的 PDF 文档
                new_page.show_pdf_page(
                    fitz.Rect(0, 0, page.rect.width, h2 - h1), pdf_document, page_number
                )
                # new_page.show_pdf_page(fitz.Rect(0, 0, page.rect.width, h2), pdf_document, page_number,
                #                        clip=clip_rect)
                # 保存新的 PDF 文档
    new_pdf_bytesio = io.BytesIO()  # 变量赋值
    new_pdf_document.save(new_pdf_bytesio)
    new_pdf_document.save("./res.pdf")
    new_pdf_bytes = new_pdf_bytesio.getvalue()  # 变量赋值
    # 关闭新的 PDF 文档
    new_pdf_document.close()  # 关闭文件句柄释放资源
    # del new_pdf_document
    # 关闭原始 PDF 文件
    pdf_document.close()  # 关闭文件句柄释放资源
    new_pdf_bytesio.close()  # 关闭文件句柄释放资源
    return new_pdf_bytes  # 返回结果


def crop_and_save_pdf_opt(  # 定义函数
    input_path,
    output_path,
    split_config,
    ocr_data,
    img_list,
    http_url,
    up_cut_str,
    down_cut_str,
):
    """
    安全的PDF剪裁和保存函数，修复内存泄漏和页面为零问题
    """
    import gc  # 导入垃圾回收模块

    pdf_document = None  # 变量赋值
    new_pdf_document = None  # 变量赋值
    new_pdf_bytesio = None  # 变量赋值

    try:
        # 打开 PDF 文件
        pdf_document = fitz.Document(input_path)  # 变量赋值
        new_pdf_document = fitz.Document()  # 变量赋值

        up_cut_pattern = re.compile(up_cut_str) if up_cut_str else None  # 变量赋值
        down_cut_pattern = re.compile(down_cut_str) if down_cut_str else None  # 变量赋值

        ppk = []  # 变量赋值
        page_added = False  # ⭐ 追踪是否添加了页面

        for page_number in range(  # 循环遍历
            pdf_document.page_count
        ):  # 使用 page_count 而不是 pageCount
            page = pdf_document[page_number]  # 变量赋值

            # 获取页面的高度
            page_height = page.rect.height  # 变量赋值
            page_width = page.rect.width  # 变量赋值
            img = img_list[page_number]  # 变量赋值
            ocr_data_i = ocr_data[page_number]  # 变量赋值

            # 检查是否需要重新OCR
            is_ok = True  # 变量赋值
            text_all = "".join(ocr_data_i.get("text", []))  # 变量赋值

            if up_cut_pattern and not up_cut_pattern.search(text_all):  # 条件判断
                is_ok = False  # 变量赋值
            if down_cut_pattern and not down_cut_pattern.search(text_all):  # 条件判断
                is_ok = False  # 变量赋值

            if len(ocr_data_i.get("text", [])) == 0 or not is_ok:  # 条件判断
                # HTTP请求获取新的OCR数据
                try:
                    img_save = cv2.imencode(".jpg", img)[1]  # 将图像编码为指定格式
                    data_encode = np.array(img_save)  # 创建NumPy数组
                    str_encode = (  # 变量赋值
                        data_encode.tobytes()  # 转换为字节串
                    )  # ⭐ 使用 tobytes() 代替过时的 tostring()
                    b64_str_encode = base64.b64encode(str_encode).decode("utf-8")  # 对字节数据进行Base64编码

                    ratio_w_h = fix_min_len_resize((page_height, page_width), 1600)  # 变量赋值
                    headers = {"Content-Type": "application/json"}  # 变量赋值
                    payload = json.dumps({"is_pdf_parse": 0, "file": b64_str_encode})  # 将对象序列化为JSON字符串

                    response = requests.request(  # 变量赋值
                        "POST", http_url, headers=headers, data=payload, timeout=30  # 变量赋值
                    )
                    http_json_res = json.loads(response.text)  # 将JSON字符串解析为对象
                    ocr_data_i = wrap_ocr_res_by_ratio(  # 变量赋值
                        http_json_res["data"], ratio_w_h[0]
                    )

                    # 及时释放response
                    response.close()  # 关闭文件句柄释放资源
                    del response, img_save, data_encode, str_encode, b64_str_encode

                except Exception as e:
                    print(f"HTTP OCR request failed for page {page_number}: {e}")
                    continue  # 跳过此页面

            # 获取span列表
            span_list = get_span_list_merge(split_config, img, ocr_data_i, page)  # 变量赋值

            # ⭐ 验证 span_list 不为空
            if not span_list:  # 条件判断
                print(f"Warning: span_list is empty for page {page_number}")
                ppk.append([])  # 添加元素到列表
                continue

            ppk.append(span_list)  # 添加元素到列表

            # ⭐ 重要：使用新的 Document 对象副本来避免修改原始文档
            temp_doc = fitz.Document()  # 变量赋值

            for span in span_list:  # 循环遍历
                h1 = span[0]  # 变量赋值
                h2 = span[1]  # 变量赋值

                # ⭐ 验证坐标有效性
                if not (0 <= h1 < h2 <= page_height):  # 条件判断
                    print(
                        f"Invalid span coordinates for page {page_number}: h1={h1}, h2={h2}, page_height={page_height}"  # 变量赋值
                    )
                    continue

                try:
                    # ⭐ 使用临时文档避免修改原始PDF
                    temp_page = temp_doc.new_page(width=page_width, height=h2 - h1)  # 变量赋值

                    # 构建剪裁区域
                    clip_rect = fitz.Rect(0, h1, page_width, h2)  # 变量赋值

                    # 显示PDF页面的指定部分
                    temp_page.show_pdf_page(
                        fitz.Rect(0, 0, page_width, h2 - h1),
                        pdf_document,
                        page_number,
                        clip=clip_rect,  # 变量赋值
                    )

                    # 将创建的页面移到新文档中
                    new_pdf_document.insert_pdf(
                        temp_doc, from_page=temp_page.number, to_page=temp_page.number  # 变量赋值
                    )

                    page_added = True  # 变量赋值

                except Exception as e:
                    print(f"Error processing span for page {page_number}: {e}")
                    continue

            # 释放临时文档
            temp_doc.close()  # 关闭文件句柄释放资源
            del temp_doc
            gc.collect()  # 定期垃圾回收

        # ⭐ 检查是否添加了任何页面
        if not page_added or new_pdf_document.page_count == 0:  # 条件判断
            raise ValueError(
                f"No valid pages after cropping. Check split_config and coordinates validation."
            )

        # 保存PDF
        new_pdf_bytesio = io.BytesIO()  # 变量赋值
        new_pdf_document.save(new_pdf_bytesio)
        new_pdf_document.save("./res.pdf")

        new_pdf_bytes = new_pdf_bytesio.getvalue()  # 变量赋值

        return new_pdf_bytes  # 返回结果

    except Exception as e:
        print(f"Error in crop_and_save_pdf: {e}")
        raise

    finally:
        # ⭐ 确保所有资源都被释放
        try:
            if new_pdf_document:  # 条件判断
                new_pdf_document.close()  # 关闭文件句柄释放资源
            if pdf_document:  # 条件判断
                pdf_document.close()  # 关闭文件句柄释放资源
            if new_pdf_bytesio:  # 条件判断
                new_pdf_bytesio.close()  # 关闭文件句柄释放资源
        except Exception as e:
            print(f"Error closing resources: {e}")

        # 强制垃圾回收
        gc.collect()  # 手动触发垃圾回收


def get_mutipage_cut_span(up_cut_str, up_filter, down_cut_str, down_filter, ocr_data):  # 定义函数
    down_pattern = re.compile(down_cut_str)  # 变量赋值
    down_filter_pattern = re.compile(down_filter)  # 变量赋值

    up_pattern = re.compile(up_cut_str)  # 变量赋值
    up_filter_pattern = re.compile(up_filter)  # 变量赋值

    is_double = False  # 变量赋值
    if down_cut_str.replace(" ", "") == "":  # 条件判断
        is_double = False  # 变量赋值
    else:  # 否则执行
        for ocr_data_i in ocr_data:  # 循环遍历
            for line_text in ocr_data_i["text"]:  # 循环遍历
                dt1 = down_pattern.search(line_text.replace(" ", ""))  # 变量赋值
                dt2 = down_filter_pattern.search(line_text.replace(" ", ""))  # 变量赋值
                if (not dt1 is None) and dt2 is None:  # 条件判断
                    is_double = True  # 变量赋值

    span_list = []  # 变量赋值
    if is_double == True:  # 条件判断
        up_match_list = []  # 变量赋值
        down_match_list = []  # 变量赋值
        for i, ocr_data_i in enumerate(ocr_data):  # 循环遍历
            for line_text in ocr_data_i["text"]:  # 循环遍历
                dt1 = up_pattern.search(line_text.replace(" ", ""))  # 变量赋值
                dt2 = up_filter_pattern.search(line_text.replace(" ", ""))  # 变量赋值
                if (not dt1 is None) and (dt2 is None):  # 条件判断
                    up_match_list.append(i)  # 添加元素到列表

        for i, ocr_data_i in enumerate(ocr_data):  # 循环遍历
            for line_text in ocr_data_i["text"]:  # 循环遍历
                dt1 = down_pattern.search(line_text.replace(" ", ""))  # 变量赋值
                dt2 = down_filter_pattern.search(line_text.replace(" ", ""))  # 变量赋值
                if (not dt1 is None) and (dt2 is None):  # 条件判断
                    down_match_list.append(i)  # 添加元素到列表
        for k in up_match_list:  # 循环遍历
            for p in down_match_list:  # 循环遍历
                if p >= k:  # 条件判断
                    span_list.append((k, p))  # 添加元素到列表
                    break
    else:  # 否则执行
        up_match_list = []  # 变量赋值
        for i, ocr_data_i in enumerate(ocr_data):  # 循环遍历
            for line_text in ocr_data_i["text"]:  # 循环遍历
                dt1 = up_pattern.search(line_text.replace(" ", ""))  # 变量赋值
                dt2 = up_filter_pattern.search(line_text.replace(" ", ""))  # 变量赋值
                if (not dt1 is None) and (dt2 is None):  # 条件判断
                    up_match_list.append(i)  # 添加元素到列表

        for d, k in enumerate(up_match_list):  # 循环遍历
            if d == len(up_match_list) - 1:  # 条件判断
                span_list.append((k, len(ocr_data) - 1))  # 添加元素到列表
            else:  # 否则执行
                span_list.append((up_match_list[d], up_match_list[d + 1]))  # 添加元素到列表

    return span_list  # 返回结果


def crop_and_save_pdf_multipage(  # 定义函数
    input_path,
    output_path,
    split_config,
    ocr_data,
    img_list,
    http_url,
    up_cut_str,
    down_cut_str,
):
    # 打开 PDF 文件
    pdf_document = fitz.Document(input_path)  # 变量赋值
    # up_cut_pattern = re.compile(up_cut_str)
    # down_cut_pattern = re.compile(down_cut_str)
    # 获取指定页数的页面对象
    up_filter = split_config["upcut_filter"]  # 变量赋值
    down_filter = split_config["downcut_filter"]  # 变量赋值
    span_list = get_mutipage_cut_span(  # 变量赋值
        up_cut_str, up_filter, down_cut_str, down_filter, ocr_data
    )
    res_list = []  # 变量赋值
    for lk, span_t in enumerate(span_list):  # 循环遍历
        new_pdf_document = fitz.Document()  # 变量赋值
        for t in range(span_t[0], span_t[1] + 1):  # 循环遍历
            page = pdf_document[t]  # 变量赋值
            new_page = new_pdf_document.new_page(  # 变量赋值
                width=page.rect.width, height=page.rect.height  # 变量赋值
            )
            new_page.show_pdf_page(
                fitz.Rect(0, 0, page.rect.width, page.rect.height), pdf_document, t
            )
        new_pdf_bytesio = io.BytesIO()  # 变量赋值
        # new_pdf_document.save('/mnt/e/项目/天赐材料/ppd9_{}.pdf'.format(lk))
        new_pdf_document.save(new_pdf_bytesio)
        new_pdf_bytes = new_pdf_bytesio.getvalue()  # 变量赋值
        # new_pdf_document.close()
        bs64_str = base64.b64encode(new_pdf_bytes).decode("utf-8")  # 对字节数据进行Base64编码
        res_list.append(bs64_str)  # 添加元素到列表
    return res_list  # 返回结果


if __name__ == "__main__":  # 条件判断
    pass
    # 输入PDF文件路径
    # input_pdf_path = "/mnt/e/项目/东航物流/东航物流-客户自有RPA-第三阶段/财务公司/回单文件/NC打印.pdf"
    # # 输出PDF文件路径
    # output_pdf_path = "/mnt/e/项目/东航物流/东航物流-客户自有RPA-第三阶段/财务公司/回单文件/output.pdf"
    # pattern_str = '有限责任公司付款凭证'
    # f = open(input_pdf_path, 'rb')
    # img_list = pdf2images_fit(f.read())
    # ocr_data, shape_list = pdf_process.get_ocr(input_pdf_path)
    # # 设置裁剪高度（在这里设置为特定高度，你可以根据需求修改）
    #
    # # 调用裁剪函数
    # b_str = crop_and_save_pdf(input_pdf_path, output_pdf_path, pattern_str, ocr_data, img_list)
