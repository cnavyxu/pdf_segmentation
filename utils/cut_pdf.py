import pdfplumber
import pdf_process_pymu
import pdf_process
import fitz
import cv2
import numpy as np
import re
import io
import requests
import json
import base64
import copy


def pdf2images_fit(b_str_image):
    pdf_doc = fitz.open(stream=b_str_image, filetype="bytes")
    img_list = []
    for i in range(pdf_doc.pageCount):
        page = pdf_doc[i]
        zoom_x = zoom_y = 1.7
        mat = fitz.Matrix(zoom_x, zoom_y)
        pix = page.getPixmap(matrix=mat)
        tt = pix.tobytes()
        img_decode = cv2.imdecode(np.frombuffer(tt, np.uint8), cv2.IMREAD_COLOR)
        img_list.append(img_decode)
    return img_list


def wrap_h(page_h, d):
    if d <= 0:
        return 0
    if d >= page_h:
        return int(page_h)
    return int(d)


def generate_span(split_list, have_pix_max, page_h, up_offset, down_offset):
    res = []
    for i, plit_d in enumerate(split_list):
        if i == len(split_list) - 1:
            res.append(
                (
                    wrap_h(page_h, split_list[i] - up_offset),
                    wrap_h(page_h, have_pix_max + down_offset),
                )
            )
        else:
            res.append(
                (
                    wrap_h(page_h, split_list[i] - up_offset),
                    wrap_h(page_h, split_list[i + 1]) - up_offset,
                )
            )
    return res


def get_span_list_single(
    pattern_str, img, ocr_data, page, up_offset, down_offset, split_config
):
    pattern = re.compile(pattern_str)
    split_list = []
    for i, text in enumerate(ocr_data["text"]):
        dt = pattern.search(text.replace(" ", ""))
        if not dt is None:

            fu_pattern = re.compile(split_config["upcut_filter"])
            if not fu_pattern.search(text) is None:
                continue
            up_h = ocr_data["blob"][i][0]["box"][0][1]  # 取 匹配字符串box 上部
            split_list.append(up_h)
    page_h, page_w = page.rect[3], page.rect[2]
    img = cv2.resize(img, (int(page_w), int(page_h)))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dd = np.where(img_gray < 100)
    have_pix_max = np.max(dd[0])
    span_list = generate_span(split_list, have_pix_max, page_h, up_offset, down_offset)
    return span_list


def get_span_list_double(
    page, up_cut_str, down_cut_str, ocr_data, up_offset, down_offset, split_config
):
    up_pattern = re.compile(up_cut_str)
    down_pattern = re.compile(down_cut_str)
    up_tmp_list = []
    down_tmp_list = []
    for i, text in enumerate(ocr_data["text"]):
        if not up_pattern.search(text.replace(" ", "")) is None:
            if split_config["upcut_filter"].replace(" ", "") == "":
                pass
            else:
                fu_pattern = re.compile(split_config["upcut_filter"])
                if not fu_pattern.search(text) is None:
                    continue
            up_tmp_list.append(i)
        if not down_pattern.search(text.replace(" ", "")) is None:
            if split_config["downcut_filter"].replace(" ", "") == "":
                pass
            else:
                fd_pattern = re.compile(split_config["downcut_filter"])
                if not fd_pattern.search(text) is None:
                    continue
            down_tmp_list.append(i)
    tmp_merge_list = []
    for k, p in enumerate(up_tmp_list):
        if k == len(up_tmp_list) - 1:
            next_ind = 10000
        else:
            next_ind = up_tmp_list[k + 1]
        last_q = None
        for q_k, q in enumerate(down_tmp_list):
            if q >= next_ind:
                break
            if q > p:
                last_q = q
        if not last_q is None:
            tmp_merge_list.append((p, last_q))

    span_list = []
    page_h, page_w = page.rect[3], page.rect[2]
    for tt_span in tmp_merge_list:
        uup = tt_span[0]
        ddp = tt_span[1]
        uup_point = wrap_h(page_h, ocr_data["blob"][uup][0]["box"][0][1] - up_offset)

        ddp_point = wrap_h(page_h, ocr_data["blob"][ddp][0]["box"][2][1] + down_offset)
        span_list.append((uup_point, ddp_point))
    return span_list


def get_span_list_merge(split_config, img, ocr_data, page):
    upcut_str = split_config["upcut_str"]
    downcut_str = split_config["downcut_str"]
    up_offset = split_config["up_offset"]
    down_offset = split_config["down_offset"]
    if len(downcut_str.replace(" ", "")) > 0:
        span_list = get_span_list_double(
            page, upcut_str, downcut_str, ocr_data, up_offset, down_offset, split_config
        )
    else:
        span_list = get_span_list_single(
            upcut_str, img, ocr_data, page, up_offset, down_offset, split_config
        )
    return span_list


def fix_min_len_resize(x_shape, min_l, max_len=2500):
    h, w = x_shape
    ratio = float(min_l) / min(h, w)
    ratio = min(max_len / max(h, w), ratio)
    new_h, new_w = int(ratio * h), int(ratio * w)
    res_ratio = (float(w) / new_w, float(h) / new_h)
    return res_ratio


def wrap_ocr_res_by_ratio(ocr_data, ratio):
    res = copy.deepcopy(ocr_data)
    for one_line_blob in res["blob"]:
        for blob in one_line_blob:
            box = blob["box"]
            box = np.asarray(box)
            box = box * ratio
            blob["box"] = box.tolist()

    for one_line_span in res["span"]:
        for k, span_k in enumerate(one_line_span):
            new_span = [int(span_k[0] * ratio), int(span_k[1] * ratio)]
            one_line_span[k] = new_span
    return res


def crop_and_save_pdf(
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
    pdf_document = fitz.Document(input_path)
    new_pdf_document = fitz.Document()
    up_cut_pattern = re.compile(up_cut_str)
    down_cut_pattern = re.compile(down_cut_str)
    # 获取指定页数的页面对象
    ppk = []
    for page_number in range(pdf_document.pageCount):
        page = pdf_document[page_number]  # 页数从0开始

        # 获取页面的高度
        page_height = page.rect.height
        page_width = page.rect.width
        img = img_list[page_number]
        ocr_data_i = ocr_data[page_number]
        # 如果指定的高度范围有效，则进行裁剪
        is_ok = True
        text_all = "".join(ocr_data_i["text"])
        if not up_cut_str == "":
            dt = up_cut_pattern.search(text_all)
            if dt is None:
                is_ok = False
        if not down_cut_str == "":
            dt = down_cut_pattern.search(text_all)
            if dt is None:
                is_ok = False
        if len(ocr_data_i["text"]) == 0 or is_ok == False:
            img_save = cv2.imencode(".jpg", img)[1]
            data_encode = np.array(img_save)
            str_encode = data_encode.tostring()
            b64_str_encode = base64.b64encode(str_encode).decode("utf-8")
            ratio_w_h = fix_min_len_resize((page_height, page_width), 1600)
            headers = {"Content-Type": "application/json"}
            payload = json.dumps({"is_pdf_parse": 0, "file": b64_str_encode})
            response = requests.request("POST", http_url, headers=headers, data=payload)
            http_json_res = json.loads(response.text)
            ocr_data_i = wrap_ocr_res_by_ratio(http_json_res["data"], ratio_w_h[0])
        span_list = get_span_list_merge(split_config, img, ocr_data_i, page)
        ppk.append(span_list)
        for span in span_list:
            h1 = span[0]
            h2 = span[1]

            if 0 <= h1 <= h2 <= page_height:
                # 构建一个矩形区域对象，定义裁剪区域
                clip_rect = fitz.Rect(0, h1, page.rect.width, h2)
                # 裁剪页面
                page.set_cropbox(clip_rect)
                new_page = new_pdf_document.new_page(
                    width=page.rect.width, height=h2 - h1
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
    new_pdf_bytesio = io.BytesIO()
    new_pdf_document.save(new_pdf_bytesio)
    new_pdf_document.save("./res.pdf")
    new_pdf_bytes = new_pdf_bytesio.getvalue()
    # 关闭新的 PDF 文档
    new_pdf_document.close()
    # del new_pdf_document
    # 关闭原始 PDF 文件
    pdf_document.close()
    new_pdf_bytesio.close()
    return new_pdf_bytes


def crop_and_save_pdf_opt(
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
    import gc

    pdf_document = None
    new_pdf_document = None
    new_pdf_bytesio = None

    try:
        # 打开 PDF 文件
        pdf_document = fitz.Document(input_path)
        new_pdf_document = fitz.Document()

        up_cut_pattern = re.compile(up_cut_str) if up_cut_str else None
        down_cut_pattern = re.compile(down_cut_str) if down_cut_str else None

        ppk = []
        page_added = False  # ⭐ 追踪是否添加了页面

        for page_number in range(
            pdf_document.page_count
        ):  # 使用 page_count 而不是 pageCount
            page = pdf_document[page_number]

            # 获取页面的高度
            page_height = page.rect.height
            page_width = page.rect.width
            img = img_list[page_number]
            ocr_data_i = ocr_data[page_number]

            # 检查是否需要重新OCR
            is_ok = True
            text_all = "".join(ocr_data_i.get("text", []))

            if up_cut_pattern and not up_cut_pattern.search(text_all):
                is_ok = False
            if down_cut_pattern and not down_cut_pattern.search(text_all):
                is_ok = False

            if len(ocr_data_i.get("text", [])) == 0 or not is_ok:
                # HTTP请求获取新的OCR数据
                try:
                    img_save = cv2.imencode(".jpg", img)[1]
                    data_encode = np.array(img_save)
                    str_encode = (
                        data_encode.tobytes()
                    )  # ⭐ 使用 tobytes() 代替过时的 tostring()
                    b64_str_encode = base64.b64encode(str_encode).decode("utf-8")

                    ratio_w_h = fix_min_len_resize((page_height, page_width), 1600)
                    headers = {"Content-Type": "application/json"}
                    payload = json.dumps({"is_pdf_parse": 0, "file": b64_str_encode})

                    response = requests.request(
                        "POST", http_url, headers=headers, data=payload, timeout=30
                    )
                    http_json_res = json.loads(response.text)
                    ocr_data_i = wrap_ocr_res_by_ratio(
                        http_json_res["data"], ratio_w_h[0]
                    )

                    # 及时释放response
                    response.close()
                    del response, img_save, data_encode, str_encode, b64_str_encode

                except Exception as e:
                    print(f"HTTP OCR request failed for page {page_number}: {e}")
                    continue  # 跳过此页面

            # 获取span列表
            span_list = get_span_list_merge(split_config, img, ocr_data_i, page)

            # ⭐ 验证 span_list 不为空
            if not span_list:
                print(f"Warning: span_list is empty for page {page_number}")
                ppk.append([])
                continue

            ppk.append(span_list)

            # ⭐ 重要：使用新的 Document 对象副本来避免修改原始文档
            temp_doc = fitz.Document()

            for span in span_list:
                h1 = span[0]
                h2 = span[1]

                # ⭐ 验证坐标有效性
                if not (0 <= h1 < h2 <= page_height):
                    print(
                        f"Invalid span coordinates for page {page_number}: h1={h1}, h2={h2}, page_height={page_height}"
                    )
                    continue

                try:
                    # ⭐ 使用临时文档避免修改原始PDF
                    temp_page = temp_doc.new_page(width=page_width, height=h2 - h1)

                    # 构建剪裁区域
                    clip_rect = fitz.Rect(0, h1, page_width, h2)

                    # 显示PDF页面的指定部分
                    temp_page.show_pdf_page(
                        fitz.Rect(0, 0, page_width, h2 - h1),
                        pdf_document,
                        page_number,
                        clip=clip_rect,
                    )

                    # 将创建的页面移到新文档中
                    new_pdf_document.insert_pdf(
                        temp_doc, from_page=temp_page.number, to_page=temp_page.number
                    )

                    page_added = True

                except Exception as e:
                    print(f"Error processing span for page {page_number}: {e}")
                    continue

            # 释放临时文档
            temp_doc.close()
            del temp_doc
            gc.collect()  # 定期垃圾回收

        # ⭐ 检查是否添加了任何页面
        if not page_added or new_pdf_document.page_count == 0:
            raise ValueError(
                f"No valid pages after cropping. Check split_config and coordinates validation."
            )

        # 保存PDF
        new_pdf_bytesio = io.BytesIO()
        new_pdf_document.save(new_pdf_bytesio)
        new_pdf_document.save("./res.pdf")

        new_pdf_bytes = new_pdf_bytesio.getvalue()

        return new_pdf_bytes

    except Exception as e:
        print(f"Error in crop_and_save_pdf: {e}")
        raise

    finally:
        # ⭐ 确保所有资源都被释放
        try:
            if new_pdf_document:
                new_pdf_document.close()
            if pdf_document:
                pdf_document.close()
            if new_pdf_bytesio:
                new_pdf_bytesio.close()
        except Exception as e:
            print(f"Error closing resources: {e}")

        # 强制垃圾回收
        gc.collect()


def get_mutipage_cut_span(up_cut_str, up_filter, down_cut_str, down_filter, ocr_data):
    down_pattern = re.compile(down_cut_str)
    down_filter_pattern = re.compile(down_filter)

    up_pattern = re.compile(up_cut_str)
    up_filter_pattern = re.compile(up_filter)

    is_double = False
    if down_cut_str.replace(" ", "") == "":
        is_double = False
    else:
        for ocr_data_i in ocr_data:
            for line_text in ocr_data_i["text"]:
                dt1 = down_pattern.search(line_text.replace(" ", ""))
                dt2 = down_filter_pattern.search(line_text.replace(" ", ""))
                if (not dt1 is None) and dt2 is None:
                    is_double = True

    span_list = []
    if is_double == True:
        up_match_list = []
        down_match_list = []
        for i, ocr_data_i in enumerate(ocr_data):
            for line_text in ocr_data_i["text"]:
                dt1 = up_pattern.search(line_text.replace(" ", ""))
                dt2 = up_filter_pattern.search(line_text.replace(" ", ""))
                if (not dt1 is None) and (dt2 is None):
                    up_match_list.append(i)

        for i, ocr_data_i in enumerate(ocr_data):
            for line_text in ocr_data_i["text"]:
                dt1 = down_pattern.search(line_text.replace(" ", ""))
                dt2 = down_filter_pattern.search(line_text.replace(" ", ""))
                if (not dt1 is None) and (dt2 is None):
                    down_match_list.append(i)
        for k in up_match_list:
            for p in down_match_list:
                if p >= k:
                    span_list.append((k, p))
                    break
    else:
        up_match_list = []
        for i, ocr_data_i in enumerate(ocr_data):
            for line_text in ocr_data_i["text"]:
                dt1 = up_pattern.search(line_text.replace(" ", ""))
                dt2 = up_filter_pattern.search(line_text.replace(" ", ""))
                if (not dt1 is None) and (dt2 is None):
                    up_match_list.append(i)

        for d, k in enumerate(up_match_list):
            if d == len(up_match_list) - 1:
                span_list.append((k, len(ocr_data) - 1))
            else:
                span_list.append((up_match_list[d], up_match_list[d + 1]))

    return span_list


def crop_and_save_pdf_multipage(
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
    pdf_document = fitz.Document(input_path)
    # up_cut_pattern = re.compile(up_cut_str)
    # down_cut_pattern = re.compile(down_cut_str)
    # 获取指定页数的页面对象
    up_filter = split_config["upcut_filter"]
    down_filter = split_config["downcut_filter"]
    span_list = get_mutipage_cut_span(
        up_cut_str, up_filter, down_cut_str, down_filter, ocr_data
    )
    res_list = []
    for lk, span_t in enumerate(span_list):
        new_pdf_document = fitz.Document()
        for t in range(span_t[0], span_t[1] + 1):
            page = pdf_document[t]
            new_page = new_pdf_document.new_page(
                width=page.rect.width, height=page.rect.height
            )
            new_page.show_pdf_page(
                fitz.Rect(0, 0, page.rect.width, page.rect.height), pdf_document, t
            )
        new_pdf_bytesio = io.BytesIO()
        # new_pdf_document.save('/mnt/e/项目/天赐材料/ppd9_{}.pdf'.format(lk))
        new_pdf_document.save(new_pdf_bytesio)
        new_pdf_bytes = new_pdf_bytesio.getvalue()
        # new_pdf_document.close()
        bs64_str = base64.b64encode(new_pdf_bytes).decode("utf-8")
        res_list.append(bs64_str)
    return res_list


if __name__ == "__main__":
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
