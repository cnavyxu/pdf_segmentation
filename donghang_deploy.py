"""
银行对账单智能解析系统 - 主部署服务
功能：提供多家银行对账单的 PDF 解析、数据提取、表格识别等服务
作者：算法团队
"""

import base64
import logging

import numpy as np
from flask import request, Flask, jsonify
from io import StringIO
from flask_cors import *

# from gevent import pywsgi
from collections import defaultdict
import pandas as pd
import json
import requests
import os
import re
import hashlib
from utils import formal_exact, huaqiao_utils
import tempfile
import time
import sys
import fitz
from utils import cut_pdf
import extract_meixi_utils, extract_huaqi_utils, extract_huifeng, extract_huaqiao
import pdf_process, extract_wuliu_huaqi, extract_deyizhi, extract_taiwan, extract_bmo
import pdf_process_pymu, extract_sanjing_duizhang, extract_zhaoshang_utils, extract_wuliu_huaqi2
import extract_sanjin_mingxi, extract_huifeng_en, extract_mogeng, extract_mogeng_en
import img2pdf
import pypdfium2 as pdfium
import gc
from utils.table_predict.predict import table_predict_main
from utils import classify_utils
import cv2
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app, supports_credentials=True)  # 启用跨域支持
app.config["JSON_AS_ASCII"] = False  # 支持中文 JSON 响应
port = 3019  # int(sys.argv[1])  # 从命令行参数获取端口号

# 初始化表格识别模型
table_model = table_predict_main.TABLE(0, "model/best.onnx", "model/table_line.onnx")


def pdf2images_fit(b_str_image, ratio=1.7):
    """
    将 PDF 文件转换为图像列表

    参数:
        b_str_image: PDF 文件的字节流
        ratio: 缩放比例，默认 1.7，用于控制输出图像的分辨率

    返回:
        img_list: 图像列表，每个元素为一页的 numpy 数组
    """
    pdf_doc = fitz.open(stream=b_str_image, filetype="bytes")
    img_list = []
    for i in range(pdf_doc.pageCount):
        page = pdf_doc[i]
        zoom_x = zoom_y = ratio  # 设置缩放比例
        mat = fitz.Matrix(zoom_x, zoom_y)  # 创建变换矩阵
        pix = page.getPixmap(matrix=mat)  # 渲染页面为像素图
        tt = pix.tobytes()  # 转换为字节流
        img_decode = cv2.imdecode(np.frombuffer(tt, np.uint8), cv2.IMREAD_COLOR)
        img_list.append(img_decode)
    fitz.TOOLS.store_shrink(100)  # 释放内存
    return img_list


def create_id():
    """
    生成唯一 ID

    返回:
        str: MD5 哈希值，基于当前时间戳生成
    """
    m = hashlib.md5(str(time.perf_counter()).encode("utf-8"))
    return m.hexdigest()


def bs64toimg(str_image):
    """
    将 Base64 编码的字符串转换为图像

    参数:
        str_image: Base64 编码的图像字符串

    返回:
        img_decode: OpenCV 图像对象（numpy 数组）
    """
    b_str_image = str_image.encode("utf-8")
    b_str_image = base64.b64decode(b_str_image)
    nparr = np.fromstring(b_str_image, np.uint8)
    img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_decode


class NumpyEncoder(json.JSONEncoder):
    """
    自定义 JSON 编码器，用于处理 NumPy 数组
    将 NumPy 数组转换为 Python 列表，以便 JSON 序列化
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def ronghui_process(res, config):
    """
    融汇处理流程 - 用于华侨银行等特定格式的数据提取

    参数:
        res: OCR 识别结果
        config: 提取配置，包含字段定义和提取规则

    返回:
        final_res: 最终提取结果，包含所有字段的值和位置信息
    """
    # 获取坐标偏移和锚点
    ratio, anchor_ind = formal_exact.get_x_y_shift(res, config)
    # 提取字段值
    rm_data_json, data_json_copy = formal_exact.exact_value(config, res)
    # 验证提取结果
    rm_data_json = formal_exact.assert_rm_data_json(rm_data_json, data_json_copy)
    # 主提取流程
    exact_result, config_res = formal_exact.main_exact_process(config, rm_data_json)
    # 获取字段位置映射
    word_position_map = formal_exact.get_position_word_map(data_json_copy, config_res)
    # 封装最终结果
    final_res = formal_exact.wrap_final_result(
        exact_result, config_res, ratio[0], word_position_map
    )
    return final_res


@app.route("/huaqiao", methods=["POST", "GET"])
def post_Data_Cetifas_huaqiao():
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    save_dir = request.form.get("save_dir")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)
    res_form, iih = extract_huaqiao.extract_all(ocr_data, shape_list)
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)
    f_obj.seek(0)
    b_str_pdf = f_obj.read()
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)
    for p in range(len(res_form)):
        for q in range(len(res_form[p]["data"])):
            pained_img = extract_meixi_utils.render_one_item(
                imgs, res_form, shape_list, iih, p, q, 2
            )
            # save_id = create_id()
            # cv2.imwrite(save_path_i, pained_img)
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))
            img_save = cv2.imencode(".jpg", pained_img)[1]
            data_encode = np.array(img_save)
            str_encode = data_encode.tostring()
            pdf_bytes = img2pdf.convert(str_encode)
            fw = open(save_path_i, "wb")
            fw.write(pdf_bytes)

            res_form[p]["data"][q]["img_path"] = save_path_i
    itemss = ronghui_process(ocr_data[0], huaqiao_utils.temp_json[0])
    gc.collect()
    res = {
        "data": {
            "item": itemss,
            "ocr_data": ocr_data_ss,
            "head_h": iih,
            "form": res_form,
            "shape_list": shape_list,
        },
        "code": 200,
        "message": "ok",
    }
    return jsonify(res)


@app.route("/meixi", methods=["POST", "GET"])
def post_Data_Cetifas():
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    save_dir = request.form.get("save_dir")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)
    res_form, iih = extract_meixi_utils.extract_all(ocr_data, shape_list)
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)
    f_obj.seek(0)
    b_str_pdf = f_obj.read()
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)
    for p in range(len(res_form)):
        for q in range(len(res_form[p]["data"])):
            pained_img = extract_meixi_utils.render_one_item(
                imgs, res_form, shape_list, iih, p, q, 2
            )
            # save_id = create_id()
            # cv2.imwrite(save_path_i, pained_img)
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))
            img_save = cv2.imencode(".jpg", pained_img)[1]
            data_encode = np.array(img_save)
            str_encode = data_encode.tostring()
            pdf_bytes = img2pdf.convert(str_encode)
            fw = open(save_path_i, "wb")
            fw.write(pdf_bytes)

            res_form[p]["data"][q]["img_path"] = save_path_i
    itemss = extract_meixi_utils.process_biaotou(ocr_data)
    gc.collect()
    res = {
        "data": {
            "item": itemss,
            "ocr_data": ocr_data_ss,
            "head_h": iih,
            "form": res_form,
            "shape_list": shape_list,
        },
        "code": 200,
        "message": "ok",
    }
    return jsonify(res)


@app.route("/cut_pdf", methods=["POST", "GET"])
def post_Data_Cetifas_cut():
    logging.info(f"start 解析参数")
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    b_str = f_obj.read()
    upcut_str = request.form.get("upcut_str")
    downcut_str = request.form.get("downcut_str")
    up_offset = int(request.form.get("up_offset"))
    down_offset = int(request.form.get("down_offset"))
    upcut_filter = request.form.get("upcut_filter")
    downcut_filter = request.form.get("downcut_filter")
    http_url = request.form.get("ocr_url")
    split_config = {
        "upcut_str": upcut_str,
        "downcut_str": downcut_str,
        "up_offset": up_offset,
        "down_offset": down_offset,
        "upcut_filter": upcut_filter,
        "downcut_filter": downcut_filter,
    }
    logging.info(f"start 裁剪")
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
        temp_file.write(b_str)
        ocr_data, shape_list = pdf_process.get_ocr_not_has_blank(temp_file.name)
        img_list = pdf2images_fit(b_str)
        b_str = cut_pdf.crop_and_save_pdf(
            temp_file.name,
            None,
            split_config,
            ocr_data,
            img_list,
            http_url,
            upcut_str,
            downcut_str,
        )
        bs64_str = base64.b64encode(b_str).decode("utf-8")
        res = {"code": 200, "message": "ok", "data": bs64_str}

    return jsonify(res)


@app.route("/cut_pdf_mutipage", methods=["POST", "GET"])
def post_Data_Cetifas_cut_multipage():
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    b_str = f_obj.read()
    upcut_str = request.form.get("upcut_str").replace(" ", "")
    downcut_str = request.form.get("downcut_str").replace(" ", "")
    up_offset = int(request.form.get("up_offset"))
    down_offset = int(request.form.get("down_offset"))
    upcut_filter = request.form.get("upcut_filter").replace(" ", "")
    downcut_filter = request.form.get("downcut_filter").replace(" ", "")
    if downcut_filter == "":
        downcut_filter = "占位符"
    if upcut_filter == "":
        upcut_filter = "占位符"
    http_url = request.form.get("ocr_url")
    split_config = {
        "upcut_str": upcut_str,
        "downcut_str": downcut_str,
        "up_offset": up_offset,
        "down_offset": down_offset,
        "upcut_filter": upcut_filter,
        "downcut_filter": downcut_filter,
    }

    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
        temp_file.write(b_str)
        ocr_data, shape_list = pdf_process.get_ocr_not_has_blank(temp_file.name)
        img_list = pdf2images_fit(b_str)
        base64_list = cut_pdf.crop_and_save_pdf_multipage(
            temp_file.name,
            None,
            split_config,
            ocr_data,
            img_list,
            http_url,
            upcut_str,
            downcut_str,
        )
        # fw = open('/mnt/e/deb.pdf', 'wb')
        # fw.write(b_str)
        # fw.close()
        # bs64_str = base64.b64encode(b_str).decode('utf-8')
        res = {"code": 200, "message": "ok", "data": base64_list}
    return jsonify(res)


@app.route("/sanjin_minxi", methods=["POST", "GET"])
def post_Data_Cetifas_sanjingminxi():
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    b_str = f_obj.read()
    save_dir = request.form.get("save_dir")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
        temp_file.write(b_str)
        ocr_data, shape_list = pdf_process_pymu.get_ocr_pymu(temp_file.name)
    res_form, iih = extract_sanjin_mingxi.extract_all(ocr_data, shape_list)
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)
    f_obj.seek(0)
    b_str_pdf = f_obj.read()
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)
    for p in range(len(res_form)):
        for q in range(len(res_form[p]["data"])):
            pained_img = extract_meixi_utils.render_one_item(
                imgs, res_form, shape_list, iih, p, q, 2
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))
            img_save = cv2.imencode(".jpg", pained_img)[1]
            data_encode = np.array(img_save)
            str_encode = data_encode.tostring()
            pdf_bytes = img2pdf.convert(str_encode)
            fw = open(save_path_i, "wb")
            fw.write(pdf_bytes)
            res_form[p]["data"][q]["img_path"] = save_path_i
    itemss = extract_sanjin_mingxi.process_biaotou(ocr_data)
    res = {
        "data": {
            "item": itemss,
            "ocr_data": ocr_data_ss,
            "head_h": iih,
            "form": res_form,
            "shape_list": shape_list,
        },
        "code": 200,
        "message": "ok",
    }
    return jsonify(res)


@app.route("/sanjin_duizhang", methods=["POST", "GET"])
def post_Data_Cetifas_sanjingduizhang():
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    save_dir = request.form.get("save_dir")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)
    res_form, iih = extract_sanjing_duizhang.extract_all(ocr_data, shape_list)
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)

    f_obj.seek(0)
    b_str_pdf = f_obj.read()
    imgs = []
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
        temp_file.write(b_str_pdf)
        pdf_pd = pdfium.PdfDocument(temp_file.name)
        for i in range(len(pdf_pd)):
            page = pdf_pd[i]
            image = page.render(scale=7).to_pil()
            imgs.append(np.asarray(image))

    # imgs = pdf2images_fit(b_str_pdf, ratio=4.0)
    for p in range(len(res_form)):
        for q in range(len(res_form[p]["data"])):
            pained_img = extract_meixi_utils.render_one_item(
                imgs, res_form, shape_list, iih, p, q, 2
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))
            img_save = cv2.imencode(".jpg", pained_img)[1]
            data_encode = np.array(img_save)
            str_encode = data_encode.tostring()
            pdf_bytes = img2pdf.convert(str_encode)
            fw = open(save_path_i, "wb")
            fw.write(pdf_bytes)
            res_form[p]["data"][q]["img_path"] = save_path_i
    itemss = extract_sanjing_duizhang.process_biaotou(ocr_data)
    res = {
        "data": {
            "item": itemss,
            "ocr_data": ocr_data_ss,
            "head_h": iih,
            "form": res_form,
            "shape_list": shape_list,
        },
        "code": 200,
        "message": "ok",
    }
    del imgs
    del pdf_pd
    return jsonify(res)


@app.route("/huaqi", methods=["POST", "GET"])
def post_Data_Cetifas_huaqi():
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    save_dir = request.form.get("save_dir")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)
    res_form, iih = extract_huaqi_utils.extract_all(ocr_data, shape_list)
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)
    f_obj.seek(0)
    b_str_pdf = f_obj.read()
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)
    for p in range(len(res_form)):
        for q in range(len(res_form[p]["data"])):
            pained_img = extract_meixi_utils.render_one_item(
                imgs, res_form, shape_list, iih, p, q, 2
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))
            img_save = cv2.imencode(".jpg", pained_img)[1]
            data_encode = np.array(img_save)
            str_encode = data_encode.tostring()
            pdf_bytes = img2pdf.convert(str_encode)
            fw = open(save_path_i, "wb")
            fw.write(pdf_bytes)
            res_form[p]["data"][q]["img_path"] = save_path_i
    res = {
        "data": {
            "item": {},
            "ocr_data": ocr_data_ss,
            "head_h": iih,
            "form": res_form,
            "shape_list": shape_list,
        },
        "code": 200,
        "message": "ok",
    }
    return jsonify(res)


@app.route("/deyizhi", methods=["POST", "GET"])
def post_Data_Cetifas_deyizhi():
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    save_dir = request.form.get("save_dir")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)
    res_form, iih = extract_deyizhi.extract_all(ocr_data, shape_list)
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)
    # res = {'data': {'ocr_data': ocr_data_ss, 'head_h': iih, 'form': res_form, 'shape_list': shape_list}, 'code': 200, 'message': 'ok'}
    f_obj.seek(0)
    b_str_pdf = f_obj.read()
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)
    for p in range(len(res_form)):
        for q in range(len(res_form[p]["data"])):
            pained_img = extract_meixi_utils.render_one_item(
                imgs, res_form, shape_list, iih, p, q, 2
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))
            img_save = cv2.imencode(".jpg", pained_img)[1]
            data_encode = np.array(img_save)
            str_encode = data_encode.tostring()
            pdf_bytes = img2pdf.convert(str_encode)
            fw = open(save_path_i, "wb")
            fw.write(pdf_bytes)
            res_form[p]["data"][q]["img_path"] = save_path_i
    itemss = extract_deyizhi.process_biaotou(ocr_data)
    res = {
        "data": {
            "item": itemss,
            "ocr_data": ocr_data_ss,
            "head_h": iih,
            "form": res_form,
            "shape_list": shape_list,
        },
        "code": 200,
        "message": "ok",
    }
    return jsonify(res)


@app.route("/bmo", methods=["POST", "GET"])
def post_Data_Cetifas_bmo():
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    save_dir = request.form.get("save_dir")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)
    res_form, iih = extract_bmo.extract_all(ocr_data, shape_list)
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)
    f_obj.seek(0)
    b_str_pdf = f_obj.read()
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)
    for p in range(len(res_form)):
        for q in range(len(res_form[p]["data"])):
            pained_img = extract_meixi_utils.render_one_item(
                imgs, res_form, shape_list, iih, p, q, 2, op_cut=73
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))
            img_save = cv2.imencode(".jpg", pained_img)[1]
            data_encode = np.array(img_save)
            str_encode = data_encode.tostring()
            pdf_bytes = img2pdf.convert(str_encode)
            fw = open(save_path_i, "wb")
            fw.write(pdf_bytes)
            res_form[p]["data"][q]["img_path"] = save_path_i
    itemss = extract_bmo.process_biaotou(ocr_data)
    res = {
        "data": {
            "item": itemss,
            "ocr_data": ocr_data_ss,
            "head_h": iih,
            "form": res_form,
            "shape_list": shape_list,
        },
        "code": 200,
        "message": "ok",
    }
    # fw = open('/home/jiangweidong/haha.json', 'w')
    # json.dump(res_form, fw)
    return jsonify(res)


# def rm_


@app.route("/huifeng_ch", methods=["POST", "GET"])
def post_Data_Cetifas_huifeng():
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    save_dir = request.form.get("save_dir")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    b_str = f_obj.read()
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
        temp_file.write(b_str)
        ocr_data, shape_list = pdf_process_pymu.get_ocr_pymu(temp_file.name)
    res_form, iih = extract_huifeng.extract_all(ocr_data, shape_list)
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)
    f_obj.seek(0)
    b_str_pdf = f_obj.read()
    imgs = pdf2images_fit(b_str_pdf, 4.0)
    for p in range(len(res_form)):
        for q in range(len(res_form[p]["data"])):
            pained_img = extract_meixi_utils.render_one_item(
                imgs, res_form, shape_list, iih, p, q, 2
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))
            img_save = cv2.imencode(".jpg", pained_img)[1]
            data_encode = np.array(img_save)
            str_encode = data_encode.tostring()
            pdf_bytes = img2pdf.convert(str_encode)
            fw = open(save_path_i, "wb")
            fw.write(pdf_bytes)
            res_form[p]["data"][q]["img_path"] = save_path_i
    itemss = extract_huifeng.process_biaotou(ocr_data)

    res = {
        "data": {
            "item": itemss,
            "ocr_data": ocr_data_ss,
            "head_h": iih,
            "form": res_form,
            "shape_list": shape_list,
        },
        "code": 200,
        "message": "ok",
    }
    return jsonify(res)


@app.route("/huifeng_en", methods=["POST", "GET"])
def post_Data_Cetifas_huifeng_english():
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    save_dir = request.form.get("save_dir")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    b_str = f_obj.read()
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
        temp_file.write(b_str)
        ocr_data, shape_list = pdf_process_pymu.get_ocr_pymu(temp_file.name)
    res_form, iih = extract_huifeng_en.extract_all(ocr_data, shape_list)
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)
    f_obj.seek(0)
    b_str_pdf = f_obj.read()
    imgs = pdf2images_fit(b_str_pdf, 4.0)
    for p in range(len(res_form)):
        for q in range(len(res_form[p]["data"])):
            pained_img = extract_meixi_utils.render_one_item(
                imgs, res_form, shape_list, iih, p, q, 2
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))
            img_save = cv2.imencode(".jpg", pained_img)[1]
            data_encode = np.array(img_save)
            str_encode = data_encode.tostring()
            pdf_bytes = img2pdf.convert(str_encode)
            fw = open(save_path_i, "wb")
            fw.write(pdf_bytes)
            res_form[p]["data"][q]["img_path"] = save_path_i
    itemss = extract_huifeng.process_biaotou_en(ocr_data)

    res = {
        "data": {
            "item": itemss,
            "ocr_data": ocr_data_ss,
            "head_h": iih,
            "form": res_form,
            "shape_list": shape_list,
        },
        "code": 200,
        "message": "ok",
    }
    return jsonify(res)


@app.route("/zhaoshang", methods=["POST", "GET"])
def post_Data_Cetifas_zhaoshang():
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    save_dir = request.form.get("save_dir")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)
    res_form, iih = extract_zhaoshang_utils.extract_all(ocr_data, shape_list)
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)
    f_obj.seek(0)
    b_str_pdf = f_obj.read()
    imgs = pdf2images_fit(b_str_pdf, ratio=5.0)
    for p in range(len(res_form)):
        for q in range(len(res_form[p]["data"])):
            pained_img = extract_meixi_utils.render_one_item(
                imgs, res_form, shape_list, iih, p, q, 2
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))
            img_save = cv2.imencode(".jpg", pained_img)[1]
            data_encode = np.array(img_save)
            str_encode = data_encode.tostring()
            pdf_bytes = img2pdf.convert(str_encode)
            fw = open(save_path_i, "wb")
            fw.write(pdf_bytes)
            res_form[p]["data"][q]["img_path"] = save_path_i
    itemss = extract_zhaoshang_utils.process_biaotou(ocr_data)
    res = {
        "data": {
            "item": itemss,
            "ocr_data": ocr_data_ss,
            "head_h": iih,
            "form": res_form,
            "shape_list": shape_list,
        },
        "code": 200,
        "message": "ok",
    }
    return jsonify(res)


@app.route("/taiwan", methods=["POST", "GET"])
def post_Data_Cetifas_taiwan():
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    save_dir = request.form.get("save_dir")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # b_str = f_obj.read()
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
    #     temp_file.write(b_str)
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)
    res_form, iih = extract_taiwan.extract_all(ocr_data, shape_list)
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)
    f_obj.seek(0)
    b_str_pdf = f_obj.read()
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)
    for p in range(len(res_form)):
        for q in range(len(res_form[p]["data"])):
            pained_img = extract_meixi_utils.render_one_item(
                imgs, res_form, shape_list, iih, p, q, 2
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))
            img_save = cv2.imencode(".jpg", pained_img)[1]
            data_encode = np.array(img_save)
            str_encode = data_encode.tostring()
            pdf_bytes = img2pdf.convert(str_encode)
            fw = open(save_path_i, "wb")
            fw.write(pdf_bytes)
            res_form[p]["data"][q]["img_path"] = save_path_i
    itemss = extract_taiwan.process_biaotou(ocr_data)
    res = {
        "data": {
            "item": itemss,
            "ocr_data": ocr_data_ss,
            "head_h": iih,
            "form": res_form,
            "shape_list": shape_list,
        },
        "code": 200,
        "message": "ok",
    }
    return jsonify(res)


@app.route("/huaqi_wuliu", methods=["POST", "GET"])
def post_Data_Cetifas_huaqi_wuliu():
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    save_dir = request.form.get("save_dir")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)
    ocr_data0 = ocr_data[0]
    class_ind = 0
    for one_line_text in ocr_data0["text"]:
        if "账户名" in one_line_text:
            class_ind = 1
            break

    op_str = "账户对账单查询"
    op_cut = 0
    for i, text in enumerate(ocr_data0["text"]):
        if op_str in text:
            op_cut = ocr_data0["blob"][i][0]["box"][2][1] + 6

    if class_ind == 0:
        res_form, iih = extract_wuliu_huaqi.extract_all(ocr_data, shape_list)
    else:
        res_form, iih = extract_wuliu_huaqi2.extract_all(ocr_data, shape_list)
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)
    f_obj.seek(0)
    b_str_pdf = f_obj.read()
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)
    for p in range(len(res_form)):
        for q in range(len(res_form[p]["data"])):
            pained_img = extract_meixi_utils.render_one_item(
                imgs, res_form, shape_list, iih, p, q, 2, op_cut
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))
            img_save = cv2.imencode(".jpg", pained_img)[1]
            data_encode = np.array(img_save)
            str_encode = data_encode.tostring()
            pdf_bytes = img2pdf.convert(str_encode)
            fw = open(save_path_i, "wb")
            fw.write(pdf_bytes)
            res_form[p]["data"][q]["img_path"] = save_path_i
    res = {
        "data": {
            "item": {},
            "ocr_data": ocr_data_ss,
            "head_h": iih,
            "form": res_form,
            "shape_list": shape_list,
        },
        "code": 200,
        "message": "ok",
    }
    return jsonify(res)


@app.route("/mogeng_ch", methods=["POST", "GET"])
def post_Data_Cetifas_mogeng_ch():
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    save_dir = request.form.get("save_dir")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)
    ocr_data0 = ocr_data[0]
    op_str = "客户参考"
    op_cut = 0
    for i, text in enumerate(ocr_data0["text"]):
        if op_str in text:
            op_cut = ocr_data0["blob"][i][0]["box"][2][1] - 25

    res_form, iih = extract_mogeng.extract_all(ocr_data, shape_list)
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)
    f_obj.seek(0)
    b_str_pdf = f_obj.read()
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)
    for p in range(len(res_form)):
        for q in range(len(res_form[p]["data"])):
            pained_img = extract_meixi_utils.render_one_item(
                imgs, res_form, shape_list, iih, p, q, 2, op_cut
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))
            img_save = cv2.imencode(".jpg", pained_img)[1]
            data_encode = np.array(img_save)
            str_encode = data_encode.tostring()
            pdf_bytes = img2pdf.convert(str_encode)
            fw = open(save_path_i, "wb")
            fw.write(pdf_bytes)
            res_form[p]["data"][q]["img_path"] = save_path_i
    itemss = extract_mogeng.process_biaotou_ch(ocr_data0)
    res = {
        "data": {
            "item": itemss,
            "ocr_data": ocr_data_ss,
            "head_h": iih,
            "form": res_form,
            "shape_list": shape_list,
        },
        "code": 200,
        "message": "ok",
    }
    return jsonify(res)


@app.route("/mogeng_en", methods=["POST", "GET"])
def post_Data_Cetifas_mogeng_en():
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    save_dir = request.form.get("save_dir")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)
    ocr_data0 = ocr_data[0]
    op_str = "Value Date Description Customer Ref".replace(" ", "").lower()
    op_cut = 0
    for i, text in enumerate(ocr_data0["text"]):
        if op_str in text:
            op_cut = ocr_data0["blob"][i][0]["box"][2][1] - 25

    res_form, iih = extract_mogeng_en.extract_all(ocr_data, shape_list)
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)
    f_obj.seek(0)
    b_str_pdf = f_obj.read()
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)
    for p in range(len(res_form)):
        for q in range(len(res_form[p]["data"])):
            pained_img = extract_meixi_utils.render_one_item(
                imgs, res_form, shape_list, iih, p, q, 2, op_cut
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))
            img_save = cv2.imencode(".jpg", pained_img)[1]
            data_encode = np.array(img_save)
            str_encode = data_encode.tostring()
            pdf_bytes = img2pdf.convert(str_encode)
            fw = open(save_path_i, "wb")
            fw.write(pdf_bytes)
            res_form[p]["data"][q]["img_path"] = save_path_i
    itemss = extract_mogeng.process_biaotou_en(ocr_data0)
    res = {
        "data": {
            "item": itemss,
            "ocr_data": ocr_data_ss,
            "head_h": iih,
            "form": res_form,
            "shape_list": shape_list,
        },
        "code": 200,
        "message": "ok",
    }
    return jsonify(res)


def fix_min_len_resize(img, min_l, max_len=2500):
    h, w = img.shape[0:2]
    ratio = float(min_l) / min(h, w)
    ratio = min(max_len / max(h, w), ratio)
    new_h, new_w = int(ratio * h), int(ratio * w)
    res_img = cv2.resize(img, (new_w, new_h))
    res_ratio = (float(w) / new_w, float(h) / new_h)
    return res_img, res_ratio


def min_max_h_span(box_list):
    min_h = 10000
    max_h = 0
    for box_i in box_list:
        max_i = np.max(box_i[:, 1])
        min_i = np.min(box_i[:, 1])
        if max_i > max_h:
            max_h = max_i

        if min_i < min_h:
            min_h = min_i

    return (min_h, max_h)


@app.route("/huifeng_yinni", methods=["POST", "GET"])
def post_Data_Cetifas_huifeng_yinni():
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    save_dir = request.form.get("save_dir")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)
    f_obj.seek(0)
    b_str = f_obj.read()
    image_list = pdf2images_fit(b_str)
    image_list = [fix_min_len_resize(img, 1600)[0] for img in image_list]
    res = []
    for p, img in enumerate(image_list):
        tmp_pa = {"page": p, "data": [], "regin": [], "shape_list": (2264, 1600)}
        contour_analyser_list = table_model.table_predict(img, ocr_data[p])
        for l, contour_analyser_i in enumerate(contour_analyser_list):
            df = pd.read_html(StringIO(contour_analyser_i.html))[0]
            # 清理数据：替换换行符并去除多余空格
            df = df.applymap(
                lambda x: x.replace("\n", " ") if isinstance(x, str) else x
            )
            df = df.applymap(
                lambda x: " ".join(str(x).split()) if isinstance(x, str) else x
            )
            df = df.fillna("")
            pos2ind_map = {}
            for i, pot in enumerate(contour_analyser_i.cell_table_coord):

                pos2ind_map[tuple(list(pot))] = i

            # 替换空字符串为 NaN
            # df.replace('', np.nan, inplace=True)

            # 转换为 NumPy 数组
            numpy_table = df.to_numpy().astype(str)
            top_region = []
            for m in range(numpy_table.shape[1]):

                ind_p = pos2ind_map[(m, 0)]

                box_p = contour_analyser_i.cell_hull_coord[ind_p]
                top_region.append(box_p)
            biaotou_span = min_max_h_span(top_region)

            img_cut_biaotou = img[0 : int(biaotou_span[1]), :, :]
            for i in range(1, numpy_table.shape[0]):
                psa = {}
                region_box_line_i = []

                cla = numpy_table[0][0].replace(" ", "")
                if cla in ["AccountLocation", "Accountcurrency"]:
                    clb = numpy_table[i][0].replace(" ", "")
                    if clb in ["PHTotal:", "SGTotal:", "THTotal:", "MYTotal:"]:
                        break

                cla = numpy_table[0][1].replace(" ", "")
                if cla in ["AccountLocation", "Accountcurrency"]:
                    clb = numpy_table[i][1].replace(" ", "")
                    if clb in ["PHTotal:", "SGTotal:", "THTotal:", "MYTotal:"]:
                        break

                for j in range(0, numpy_table.shape[1]):

                    try:
                        ind_ij = pos2ind_map[(j, i)]
                        if j == 0:
                            pass
                        else:
                            region_box_line_i.append(
                                contour_analyser_i.cell_hull_coord[ind_ij]
                            )

                        psa[numpy_table[0][j].replace(" ", "")] = numpy_table[i][j]
                    except:
                        psa[numpy_table[0][j].replace(" ", "")] = ""

                ele_span_i = min_max_h_span(region_box_line_i)
                img_cut_i = img[int(ele_span_i[0]) : int(ele_span_i[1]), :, :]
                img_merge = np.concatenate([img_cut_biaotou, img_cut_i], axis=0)
                img_save = cv2.imencode(".jpg", img_merge)[1]
                data_encode = np.array(img_save)
                str_encode = data_encode.tostring()
                pdf_bytes = img2pdf.convert(str_encode)
                psa["img_path"] = os.path.join(save_dir, "{}_{}_{}.pdf".format(p, l, i))
                fw = open(psa["img_path"], "wb")
                fw.write(pdf_bytes)
                tmp_pa["data"].append(psa)
        res.append(tmp_pa)
    all_res = {
        "data": {"form": res, "item": {}, "shape_list": [], "ocr_data": ""},
        "code": 200,
        "message": "ok",
    }
    return jsonify(all_res)


@app.route("/is_pdf_parse", methods=["POST", "GET"])
def is_pdf_parse():
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)
    ocr_data0 = ocr_data[0]
    has_ind_num = 0
    for line_text in ocr_data0["text"]:
        if len(line_text) > 0:
            has_ind_num += 1
    if has_ind_num >= 3:
        res = {"is_pdf_parse": 1, "code": 200, "message": "ok"}
    else:
        res = {"is_pdf_parse": 0, "code": 200, "message": "ok"}
    return jsonify(res)


@app.route("/render_file", methods=["POST", "GET"])
def post_Data_Cetifas_file():
    f_json = request.get_json()
    path = f_json["file"]
    res_blob = f_json["form"]
    thickness = f_json["thickness"]
    page = f_json["page"]
    line_ind = f_json["line_ind"]
    shape_list = f_json["shape_list"]
    iis = f_json["head_h"]
    f = open(path, "rb")
    b_str_pdf = f.read()
    imgs = pdf2images_fit(b_str_pdf)
    final_img = extract_meixi_utils.render_one_item(
        imgs, res_blob, shape_list, iis, page, line_ind, thickness
    )
    img_save = cv2.imencode(".jpg", final_img)[1]
    data_encode = np.array(img_save)
    str_encode = data_encode.tostring()
    b64_str_encode_render = base64.b64encode(str_encode).decode("utf-8")
    res = {"message": "ok", "code": 200, "data": b64_str_encode_render}
    return jsonify(res)


@app.route("/render", methods=["POST", "GET"])
def post_Data_Cetifas_rrr():
    f_json = request.get_json()
    bs64_pdf_str = f_json["file"]
    res_blob = f_json["form"]
    thickness = f_json["thickness"]
    page = f_json["page"]
    line_ind = f_json["line_ind"]
    shape_list = f_json["shape_list"]
    iis = f_json["head_h"]
    b_str_pdf_base64 = bs64_pdf_str.encode("utf-8")
    b_str_pdf = base64.b64decode(b_str_pdf_base64)
    imgs = pdf2images_fit(b_str_pdf)
    final_img = extract_meixi_utils.render_one_item(
        imgs, res_blob, shape_list, iis, page, line_ind, thickness
    )
    img_save = cv2.imencode(".jpg", final_img)[1]
    data_encode = np.array(img_save)
    str_encode = data_encode.tostring()
    b64_str_encode_render = base64.b64encode(str_encode).decode("utf-8")
    res = {"message": "ok", "code": 200, "data": b64_str_encode_render}
    return jsonify(res)


@app.route("/render_touming", methods=["POST", "GET"])
def post_Data_Cetifas_rrr_touming():
    f_json = request.get_json()
    bs64_pdf_str = f_json["file"]
    res_blob = f_json["form"]
    thickness = f_json["thickness"]
    page = f_json["page"]
    line_ind = f_json["line_ind"]
    shape_list = f_json["shape_list"]
    iis = f_json["head_h"]
    b_str_pdf_base64 = bs64_pdf_str.encode("utf-8")
    b_str_pdf = base64.b64decode(b_str_pdf_base64)
    imgs = pdf2images_fit(b_str_pdf)
    final_img = extract_meixi_utils.render_one_item(
        imgs, res_blob, shape_list, iis, page, line_ind, thickness
    )
    img_save = cv2.imencode(".jpg", final_img)[1]
    data_encode = np.array(img_save)
    str_encode = data_encode.tostring()
    b64_str_encode_render = base64.b64encode(str_encode).decode("utf-8")
    res = {"message": "ok", "code": 200, "data": b64_str_encode_render}
    return jsonify(res)


@app.route("/classify_pdf", methods=["POST", "GET"])
def classify_pdf():
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    config_str = request.form.get("config")
    config_json = json.loads(config_str)
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)
    class_name = classify_utils.classify_run(ocr_data, config_json)
    res = {"message": "ok", "code": 200, "data": class_name}
    return jsonify(res)


@app.route("/mogeng_merge", methods=["POST", "GET"])
def mogeng_merge():

    url = "http://127.0.0.1:{}/classify_pdf".format(port)
    payload = {
        "config": '{"page":0,"data":{"en":{"da":["Description Customer Ref"],"no":["客户参考 银行参考"]},"ch":{"da":["客户参考 银行参考"],"no":["Description Customer Ref"]}}}'
    }
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    save_dir = request.form.get("save_dir")
    files = [("file", (f_obj.name, f_obj, "application/pdf"))]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    f_obj.seek(0)
    cc = response.text
    cc = json.loads(cc)

    if cc["data"] == "ch":
        url = "http://127.0.0.1:{}/mogeng_ch".format(port)
        payload = {"save_dir": save_dir}
        files = [("file", (f_obj.name, f_obj, "application/pdf"))]
        headers = {}
        response = requests.request(
            "POST", url, headers=headers, data=payload, files=files
        )

    else:
        url = "http://127.0.0.1:{}/mogeng_en".format(port)
        payload = {"save_dir": save_dir}
        files = [("file", (f_obj.name, f_obj, "application/pdf"))]
        headers = {}
        response = requests.request(
            "POST", url, headers=headers, data=payload, files=files
        )
    cc = response.text
    cc = json.loads(cc)
    return jsonify(cc)


@app.route("/huifeng_merge", methods=["POST", "GET"])
def huifeng_merge():

    url = "http://127.0.0.1:{}/classify_pdf".format(port)
    payload = {
        "config": '{"page":0,"data":{"en":{"da":["Bank reference"],"no":["客户参考"]},"ch":{"da":["客户参考"],"no":["Bank reference"]}}}'
    }
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    save_dir = request.form.get("save_dir")
    files = [("file", (f_obj.name, f_obj, "application/pdf"))]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    f_obj.seek(0)
    cc = response.text
    cc = json.loads(cc)

    if cc["data"] == "ch":
        url = "http://127.0.0.1:{}/huifeng_ch".format(port)
        payload = {"save_dir": save_dir}
        files = [("file", (f_obj.name, f_obj, "application/pdf"))]
        headers = {}
        response = requests.request(
            "POST", url, headers=headers, data=payload, files=files
        )

    else:
        url = "http://127.0.0.1:{}/huifeng_en".format(port)
        payload = {"save_dir": save_dir}
        files = [("file", (f_obj.name, f_obj, "application/pdf"))]
        headers = {}
        response = requests.request(
            "POST", url, headers=headers, data=payload, files=files
        )
    cc = response.text
    cc = json.loads(cc)
    return jsonify(cc)


@app.route("/huifeng_cut", methods=["POST", "GET"])
def huifeng_cut():
    biyou = "AdviceType通知种类"
    f_obj_tt = request.files
    f_obj = f_obj_tt.get("file")
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)
    f_obj.seek(0)
    biyou_id_list = []
    for i, ocr_data_i in enumerate(ocr_data):
        text_all_i = "".join(ocr_data_i["raw_text"])
        text_all_i = text_all_i.replace(" ", "")
        if biyou in text_all_i:
            biyou_id_list.append(i)
    res_span = []

    for i in range(len(biyou_id_list)):
        if i + 1 < len(biyou_id_list):
            res_span.append((biyou_id_list[i], biyou_id_list[i + 1]))
        else:
            res_span.append((biyou_id_list[i], len(ocr_data)))

    res = []

    pdf_reader = PdfReader(f_obj)
    for i, span_i in enumerate(res_span):
        pdf_writer = PdfWriter()
        for j in range(span_i[0], span_i[1]):
            pdf_writer.add_page(pdf_reader.pages[j])
        output_buffer = BytesIO()
        pdf_writer.write(output_buffer)
        pdf_bytes = output_buffer.getvalue()
        output_buffer.close()
        b64_str_encode_render = base64.b64encode(pdf_bytes).decode("utf-8")
        res.append(b64_str_encode_render)
    # print('aaa')

    return {"message": "ok", "code": 200, "data": res}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, threaded=True)
    # ttt = pywsgi.WSGIServer.get_environ()
    # server = pywsgi.WSGIServer(('0.0.0.0', port), app)
    # ttt = server.get_environ()
    # server.serve_forever()
