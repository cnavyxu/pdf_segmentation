"""
银行对账单智能解析系统 - 主部署服务
功能：提供多家银行对账单的 PDF 解析、数据提取、表格识别等服务
作者：算法团队
"""

import base64  # 导入base64库用于Base64编码解码操作
import logging  # 导入日志模块用于记录程序运行信息

import numpy as np  # 导入NumPy库用于数值计算和数组操作
from flask import request, Flask, jsonify  # 从Flask框架导入Web服务所需组件
from io import StringIO  # 从io模块导入字符串流类
from flask_cors import *  # 导入Flask跨域支持扩展

# from gevent import pywsgi
from collections import defaultdict  # 导入默认字典类
import pandas as pd  # 导入Pandas数据分析库
import json  # 导入JSON数据处理库
import requests  # 导入HTTP请求库
import os  # 导入操作系统接口模块
import re  # 导入正则表达式模块
import hashlib  # 导入哈希算法库
from utils import formal_exact, huaqiao_utils
import tempfile  # 导入临时文件处理模块
import time  # 导入时间处理模块
import sys  # 导入系统特定参数和函数模块
import fitz  # 导入PyMuPDF库用于PDF处理
from utils import cut_pdf
import extract_meixi_utils, extract_huaqi_utils, extract_huifeng, extract_huaqiao
import pdf_process, extract_wuliu_huaqi, extract_deyizhi, extract_taiwan, extract_bmo
import pdf_process_pymu, extract_sanjing_duizhang, extract_zhaoshang_utils, extract_wuliu_huaqi2
import extract_sanjin_mingxi, extract_huifeng_en, extract_mogeng, extract_mogeng_en
import img2pdf  # 导入图像转PDF库
import pypdfium2 as pdfium  # 导入pypdfium2高质量PDF渲染库
import gc  # 导入垃圾回收模块
from utils.table_predict.predict import table_predict_main
from utils import classify_utils
import cv2  # 导入OpenCV计算机视觉库
from io import BytesIO  # 从io模块导入字节流类
from PyPDF2 import PdfReader, PdfWriter  # 从PyPDF2库导入PDF操作类

# 初始化 Flask 应用
app = Flask(__name__)  # 创建Flask应用实例
CORS(app, supports_credentials=True)  # 启用跨域支持
app.config["JSON_AS_ASCII"] = False  # 支持中文 JSON 响应
port = 3019  # int(sys.argv[1])  # 从命令行参数获取端口号

# 初始化表格识别模型
table_model = table_predict_main.TABLE(0, "model/best.onnx", "model/table_line.onnx")  # 变量赋值


def pdf2images_fit(b_str_image, ratio=1.7):  # 定义函数
    """
    将 PDF 文件转换为图像列表

    参数:
        b_str_image: PDF 文件的字节流
        ratio: 缩放比例，默认 1.7，用于控制输出图像的分辨率

    返回:
        img_list: 图像列表，每个元素为一页的 numpy 数组
    """
    pdf_doc = fitz.open(stream=b_str_image, filetype="bytes")  # 打开文件或资源
    img_list = []  # 变量赋值
    for i in range(pdf_doc.pageCount):  # 循环遍历
        page = pdf_doc[i]  # 变量赋值
        zoom_x = zoom_y = ratio  # 设置缩放比例
        mat = fitz.Matrix(zoom_x, zoom_y)  # 创建变换矩阵
        pix = page.getPixmap(matrix=mat)  # 渲染页面为像素图
        tt = pix.tobytes()  # 转换为字节流
        img_decode = cv2.imdecode(np.frombuffer(tt, np.uint8), cv2.IMREAD_COLOR)  # 将字节流解码为图像
        img_list.append(img_decode)  # 添加元素到列表
    fitz.TOOLS.store_shrink(100)  # 释放内存
    return img_list  # 返回结果


def create_id():  # 定义函数
    """
    生成唯一 ID

    返回:
        str: MD5 哈希值，基于当前时间戳生成
    """
    m = hashlib.md5(str(time.perf_counter()).encode("utf-8"))  # 变量赋值
    return m.hexdigest()  # 返回结果


def bs64toimg(str_image):  # 定义函数
    """
    将 Base64 编码的字符串转换为图像

    参数:
        str_image: Base64 编码的图像字符串

    返回:
        img_decode: OpenCV 图像对象（numpy 数组）
    """
    b_str_image = str_image.encode("utf-8")  # 变量赋值
    b_str_image = base64.b64decode(b_str_image)  # 对Base64字符串进行解码
    nparr = np.fromstring(b_str_image, np.uint8)  # 变量赋值
    img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 将字节流解码为图像
    return img_decode  # 返回结果


class NumpyEncoder(json.JSONEncoder):  # 定义类
    """
    自定义 JSON 编码器，用于处理 NumPy 数组
    将 NumPy 数组转换为 Python 列表，以便 JSON 序列化
    """

    def default(self, obj):  # 定义函数
        if isinstance(obj, np.ndarray):  # 条件判断
            return obj.tolist()  # 返回结果
        return json.JSONEncoder.default(self, obj)  # 返回结果


def ronghui_process(res, config):  # 定义函数
    """
    融汇处理流程 - 用于华侨银行等特定格式的数据提取

    参数:
        res: OCR 识别结果
        config: 提取配置，包含字段定义和提取规则

    返回:
        final_res: 最终提取结果，包含所有字段的值和位置信息
    """
    # 获取坐标偏移和锚点
    ratio, anchor_ind = formal_exact.get_x_y_shift(res, config)  # 变量赋值
    # 提取字段值
    rm_data_json, data_json_copy = formal_exact.exact_value(config, res)  # 变量赋值
    # 验证提取结果
    rm_data_json = formal_exact.assert_rm_data_json(rm_data_json, data_json_copy)  # 变量赋值
    # 主提取流程
    exact_result, config_res = formal_exact.main_exact_process(config, rm_data_json)  # 变量赋值
    # 获取字段位置映射
    word_position_map = formal_exact.get_position_word_map(data_json_copy, config_res)  # 变量赋值
    # 封装最终结果
    final_res = formal_exact.wrap_final_result(  # 变量赋值
        exact_result, config_res, ratio[0], word_position_map
    )
    return final_res  # 返回结果


@app.route("/huaqiao", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_huaqiao():  # 定义函数
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    save_dir = request.form.get("save_dir")  # 从表单数据获取参数
    if not os.path.exists(save_dir):  # 检查路径是否存在
        os.makedirs(save_dir)  # 创建目录结构
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)  # 变量赋值
    res_form, iih = extract_huaqiao.extract_all(ocr_data, shape_list)  # 变量赋值
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)  # 将对象序列化为JSON字符串
    f_obj.seek(0)  # 重置文件指针到开始位置
    b_str_pdf = f_obj.read()  # 读取文件内容
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)  # 变量赋值
    for p in range(len(res_form)):  # 循环遍历
        for q in range(len(res_form[p]["data"])):  # 循环遍历
            pained_img = extract_meixi_utils.render_one_item(  # 变量赋值
                imgs, res_form, shape_list, iih, p, q, 2
            )
            # save_id = create_id()
            # cv2.imwrite(save_path_i, pained_img)
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))  # 拼接文件路径
            img_save = cv2.imencode(".jpg", pained_img)[1]  # 将图像编码为指定格式
            data_encode = np.array(img_save)  # 创建NumPy数组
            str_encode = data_encode.tostring()  # 转换为字节字符串
            pdf_bytes = img2pdf.convert(str_encode)  # 变量赋值
            fw = open(save_path_i, "wb")  # 变量赋值
            fw.write(pdf_bytes)  # 写入数据到文件

            res_form[p]["data"][q]["img_path"] = save_path_i  # 变量赋值
    itemss = ronghui_process(ocr_data[0], huaqiao_utils.temp_json[0])  # 变量赋值
    gc.collect()  # 手动触发垃圾回收
    res = {  # 变量赋值
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
    return jsonify(res)  # 返回结果


@app.route("/meixi", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas():  # 定义函数
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    save_dir = request.form.get("save_dir")  # 从表单数据获取参数
    if not os.path.exists(save_dir):  # 检查路径是否存在
        os.makedirs(save_dir)  # 创建目录结构
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)  # 变量赋值
    res_form, iih = extract_meixi_utils.extract_all(ocr_data, shape_list)  # 变量赋值
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)  # 将对象序列化为JSON字符串
    f_obj.seek(0)  # 重置文件指针到开始位置
    b_str_pdf = f_obj.read()  # 读取文件内容
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)  # 变量赋值
    for p in range(len(res_form)):  # 循环遍历
        for q in range(len(res_form[p]["data"])):  # 循环遍历
            pained_img = extract_meixi_utils.render_one_item(  # 变量赋值
                imgs, res_form, shape_list, iih, p, q, 2
            )
            # save_id = create_id()
            # cv2.imwrite(save_path_i, pained_img)
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))  # 拼接文件路径
            img_save = cv2.imencode(".jpg", pained_img)[1]  # 将图像编码为指定格式
            data_encode = np.array(img_save)  # 创建NumPy数组
            str_encode = data_encode.tostring()  # 转换为字节字符串
            pdf_bytes = img2pdf.convert(str_encode)  # 变量赋值
            fw = open(save_path_i, "wb")  # 变量赋值
            fw.write(pdf_bytes)  # 写入数据到文件

            res_form[p]["data"][q]["img_path"] = save_path_i  # 变量赋值
    itemss = extract_meixi_utils.process_biaotou(ocr_data)  # 变量赋值
    gc.collect()  # 手动触发垃圾回收
    res = {  # 变量赋值
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
    return jsonify(res)  # 返回结果


@app.route("/cut_pdf", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_cut():  # 定义函数
    logging.info(f"start 解析参数")
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    b_str = f_obj.read()  # 读取文件内容
    upcut_str = request.form.get("upcut_str")  # 从表单数据获取参数
    downcut_str = request.form.get("downcut_str")  # 从表单数据获取参数
    up_offset = int(request.form.get("up_offset"))  # 从表单数据获取参数
    down_offset = int(request.form.get("down_offset"))  # 从表单数据获取参数
    upcut_filter = request.form.get("upcut_filter")  # 从表单数据获取参数
    downcut_filter = request.form.get("downcut_filter")  # 从表单数据获取参数
    http_url = request.form.get("ocr_url")  # 从表单数据获取参数
    split_config = {  # 变量赋值
        "upcut_str": upcut_str,
        "downcut_str": downcut_str,
        "up_offset": up_offset,
        "down_offset": down_offset,
        "upcut_filter": upcut_filter,
        "downcut_filter": downcut_filter,
    }
    logging.info(f"start 裁剪")
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:  # 变量赋值
        temp_file.write(b_str)  # 写入数据到文件
        ocr_data, shape_list = pdf_process.get_ocr_not_has_blank(temp_file.name)  # 变量赋值
        img_list = pdf2images_fit(b_str)  # 变量赋值
        b_str = cut_pdf.crop_and_save_pdf(  # 变量赋值
            temp_file.name,
            None,
            split_config,
            ocr_data,
            img_list,
            http_url,
            upcut_str,
            downcut_str,
        )
        bs64_str = base64.b64encode(b_str).decode("utf-8")  # 对字节数据进行Base64编码
        res = {"code": 200, "message": "ok", "data": bs64_str}  # 变量赋值

    return jsonify(res)  # 返回结果


@app.route("/cut_pdf_mutipage", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_cut_multipage():  # 定义函数
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    b_str = f_obj.read()  # 读取文件内容
    upcut_str = request.form.get("upcut_str").replace(" ", "")  # 从表单数据获取参数
    downcut_str = request.form.get("downcut_str").replace(" ", "")  # 从表单数据获取参数
    up_offset = int(request.form.get("up_offset"))  # 从表单数据获取参数
    down_offset = int(request.form.get("down_offset"))  # 从表单数据获取参数
    upcut_filter = request.form.get("upcut_filter").replace(" ", "")  # 从表单数据获取参数
    downcut_filter = request.form.get("downcut_filter").replace(" ", "")  # 从表单数据获取参数
    if downcut_filter == "":  # 条件判断
        downcut_filter = "占位符"  # 变量赋值
    if upcut_filter == "":  # 条件判断
        upcut_filter = "占位符"  # 变量赋值
    http_url = request.form.get("ocr_url")  # 从表单数据获取参数
    split_config = {  # 变量赋值
        "upcut_str": upcut_str,
        "downcut_str": downcut_str,
        "up_offset": up_offset,
        "down_offset": down_offset,
        "upcut_filter": upcut_filter,
        "downcut_filter": downcut_filter,
    }

    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:  # 变量赋值
        temp_file.write(b_str)  # 写入数据到文件
        ocr_data, shape_list = pdf_process.get_ocr_not_has_blank(temp_file.name)  # 变量赋值
        img_list = pdf2images_fit(b_str)  # 变量赋值
        base64_list = cut_pdf.crop_and_save_pdf_multipage(  # 变量赋值
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
        res = {"code": 200, "message": "ok", "data": base64_list}  # 变量赋值
    return jsonify(res)  # 返回结果


@app.route("/sanjin_minxi", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_sanjingminxi():  # 定义函数
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    b_str = f_obj.read()  # 读取文件内容
    save_dir = request.form.get("save_dir")  # 从表单数据获取参数
    if not os.path.exists(save_dir):  # 检查路径是否存在
        os.makedirs(save_dir)  # 创建目录结构
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:  # 变量赋值
        temp_file.write(b_str)  # 写入数据到文件
        ocr_data, shape_list = pdf_process_pymu.get_ocr_pymu(temp_file.name)  # 变量赋值
    res_form, iih = extract_sanjin_mingxi.extract_all(ocr_data, shape_list)  # 变量赋值
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)  # 将对象序列化为JSON字符串
    f_obj.seek(0)  # 重置文件指针到开始位置
    b_str_pdf = f_obj.read()  # 读取文件内容
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)  # 变量赋值
    for p in range(len(res_form)):  # 循环遍历
        for q in range(len(res_form[p]["data"])):  # 循环遍历
            pained_img = extract_meixi_utils.render_one_item(  # 变量赋值
                imgs, res_form, shape_list, iih, p, q, 2
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))  # 拼接文件路径
            img_save = cv2.imencode(".jpg", pained_img)[1]  # 将图像编码为指定格式
            data_encode = np.array(img_save)  # 创建NumPy数组
            str_encode = data_encode.tostring()  # 转换为字节字符串
            pdf_bytes = img2pdf.convert(str_encode)  # 变量赋值
            fw = open(save_path_i, "wb")  # 变量赋值
            fw.write(pdf_bytes)  # 写入数据到文件
            res_form[p]["data"][q]["img_path"] = save_path_i  # 变量赋值
    itemss = extract_sanjin_mingxi.process_biaotou(ocr_data)  # 变量赋值
    res = {  # 变量赋值
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
    return jsonify(res)  # 返回结果


@app.route("/sanjin_duizhang", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_sanjingduizhang():  # 定义函数
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    save_dir = request.form.get("save_dir")  # 从表单数据获取参数
    if not os.path.exists(save_dir):  # 检查路径是否存在
        os.makedirs(save_dir)  # 创建目录结构
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)  # 变量赋值
    res_form, iih = extract_sanjing_duizhang.extract_all(ocr_data, shape_list)  # 变量赋值
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)  # 将对象序列化为JSON字符串

    f_obj.seek(0)  # 重置文件指针到开始位置
    b_str_pdf = f_obj.read()  # 读取文件内容
    imgs = []  # 变量赋值
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:  # 变量赋值
        temp_file.write(b_str_pdf)  # 写入数据到文件
        pdf_pd = pdfium.PdfDocument(temp_file.name)  # 变量赋值
        for i in range(len(pdf_pd)):  # 循环遍历
            page = pdf_pd[i]  # 变量赋值
            image = page.render(scale=7).to_pil()  # 变量赋值
            imgs.append(np.asarray(image))  # 添加元素到列表

    # imgs = pdf2images_fit(b_str_pdf, ratio=4.0)
    for p in range(len(res_form)):  # 循环遍历
        for q in range(len(res_form[p]["data"])):  # 循环遍历
            pained_img = extract_meixi_utils.render_one_item(  # 变量赋值
                imgs, res_form, shape_list, iih, p, q, 2
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))  # 拼接文件路径
            img_save = cv2.imencode(".jpg", pained_img)[1]  # 将图像编码为指定格式
            data_encode = np.array(img_save)  # 创建NumPy数组
            str_encode = data_encode.tostring()  # 转换为字节字符串
            pdf_bytes = img2pdf.convert(str_encode)  # 变量赋值
            fw = open(save_path_i, "wb")  # 变量赋值
            fw.write(pdf_bytes)  # 写入数据到文件
            res_form[p]["data"][q]["img_path"] = save_path_i  # 变量赋值
    itemss = extract_sanjing_duizhang.process_biaotou(ocr_data)  # 变量赋值
    res = {  # 变量赋值
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
    return jsonify(res)  # 返回结果


@app.route("/huaqi", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_huaqi():  # 定义函数
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    save_dir = request.form.get("save_dir")  # 从表单数据获取参数
    if not os.path.exists(save_dir):  # 检查路径是否存在
        os.makedirs(save_dir)  # 创建目录结构
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)  # 变量赋值
    res_form, iih = extract_huaqi_utils.extract_all(ocr_data, shape_list)  # 变量赋值
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)  # 将对象序列化为JSON字符串
    f_obj.seek(0)  # 重置文件指针到开始位置
    b_str_pdf = f_obj.read()  # 读取文件内容
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)  # 变量赋值
    for p in range(len(res_form)):  # 循环遍历
        for q in range(len(res_form[p]["data"])):  # 循环遍历
            pained_img = extract_meixi_utils.render_one_item(  # 变量赋值
                imgs, res_form, shape_list, iih, p, q, 2
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))  # 拼接文件路径
            img_save = cv2.imencode(".jpg", pained_img)[1]  # 将图像编码为指定格式
            data_encode = np.array(img_save)  # 创建NumPy数组
            str_encode = data_encode.tostring()  # 转换为字节字符串
            pdf_bytes = img2pdf.convert(str_encode)  # 变量赋值
            fw = open(save_path_i, "wb")  # 变量赋值
            fw.write(pdf_bytes)  # 写入数据到文件
            res_form[p]["data"][q]["img_path"] = save_path_i  # 变量赋值
    res = {  # 变量赋值
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
    return jsonify(res)  # 返回结果


@app.route("/deyizhi", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_deyizhi():  # 定义函数
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    save_dir = request.form.get("save_dir")  # 从表单数据获取参数
    if not os.path.exists(save_dir):  # 检查路径是否存在
        os.makedirs(save_dir)  # 创建目录结构
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)  # 变量赋值
    res_form, iih = extract_deyizhi.extract_all(ocr_data, shape_list)  # 变量赋值
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)  # 将对象序列化为JSON字符串
    # res = {'data': {'ocr_data': ocr_data_ss, 'head_h': iih, 'form': res_form, 'shape_list': shape_list}, 'code': 200, 'message': 'ok'}
    f_obj.seek(0)  # 重置文件指针到开始位置
    b_str_pdf = f_obj.read()  # 读取文件内容
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)  # 变量赋值
    for p in range(len(res_form)):  # 循环遍历
        for q in range(len(res_form[p]["data"])):  # 循环遍历
            pained_img = extract_meixi_utils.render_one_item(  # 变量赋值
                imgs, res_form, shape_list, iih, p, q, 2
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))  # 拼接文件路径
            img_save = cv2.imencode(".jpg", pained_img)[1]  # 将图像编码为指定格式
            data_encode = np.array(img_save)  # 创建NumPy数组
            str_encode = data_encode.tostring()  # 转换为字节字符串
            pdf_bytes = img2pdf.convert(str_encode)  # 变量赋值
            fw = open(save_path_i, "wb")  # 变量赋值
            fw.write(pdf_bytes)  # 写入数据到文件
            res_form[p]["data"][q]["img_path"] = save_path_i  # 变量赋值
    itemss = extract_deyizhi.process_biaotou(ocr_data)  # 变量赋值
    res = {  # 变量赋值
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
    return jsonify(res)  # 返回结果


@app.route("/bmo", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_bmo():  # 定义函数
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    save_dir = request.form.get("save_dir")  # 从表单数据获取参数
    if not os.path.exists(save_dir):  # 检查路径是否存在
        os.makedirs(save_dir)  # 创建目录结构
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)  # 变量赋值
    res_form, iih = extract_bmo.extract_all(ocr_data, shape_list)  # 变量赋值
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)  # 将对象序列化为JSON字符串
    f_obj.seek(0)  # 重置文件指针到开始位置
    b_str_pdf = f_obj.read()  # 读取文件内容
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)  # 变量赋值
    for p in range(len(res_form)):  # 循环遍历
        for q in range(len(res_form[p]["data"])):  # 循环遍历
            pained_img = extract_meixi_utils.render_one_item(  # 变量赋值
                imgs, res_form, shape_list, iih, p, q, 2, op_cut=73  # 变量赋值
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))  # 拼接文件路径
            img_save = cv2.imencode(".jpg", pained_img)[1]  # 将图像编码为指定格式
            data_encode = np.array(img_save)  # 创建NumPy数组
            str_encode = data_encode.tostring()  # 转换为字节字符串
            pdf_bytes = img2pdf.convert(str_encode)  # 变量赋值
            fw = open(save_path_i, "wb")  # 变量赋值
            fw.write(pdf_bytes)  # 写入数据到文件
            res_form[p]["data"][q]["img_path"] = save_path_i  # 变量赋值
    itemss = extract_bmo.process_biaotou(ocr_data)  # 变量赋值
    res = {  # 变量赋值
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
    return jsonify(res)  # 返回结果


# def rm_


@app.route("/huifeng_ch", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_huifeng():  # 定义函数
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    save_dir = request.form.get("save_dir")  # 从表单数据获取参数
    if not os.path.exists(save_dir):  # 检查路径是否存在
        os.makedirs(save_dir)  # 创建目录结构
    b_str = f_obj.read()  # 读取文件内容
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:  # 变量赋值
        temp_file.write(b_str)  # 写入数据到文件
        ocr_data, shape_list = pdf_process_pymu.get_ocr_pymu(temp_file.name)  # 变量赋值
    res_form, iih = extract_huifeng.extract_all(ocr_data, shape_list)  # 变量赋值
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)  # 将对象序列化为JSON字符串
    f_obj.seek(0)  # 重置文件指针到开始位置
    b_str_pdf = f_obj.read()  # 读取文件内容
    imgs = pdf2images_fit(b_str_pdf, 4.0)  # 变量赋值
    for p in range(len(res_form)):  # 循环遍历
        for q in range(len(res_form[p]["data"])):  # 循环遍历
            pained_img = extract_meixi_utils.render_one_item(  # 变量赋值
                imgs, res_form, shape_list, iih, p, q, 2
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))  # 拼接文件路径
            img_save = cv2.imencode(".jpg", pained_img)[1]  # 将图像编码为指定格式
            data_encode = np.array(img_save)  # 创建NumPy数组
            str_encode = data_encode.tostring()  # 转换为字节字符串
            pdf_bytes = img2pdf.convert(str_encode)  # 变量赋值
            fw = open(save_path_i, "wb")  # 变量赋值
            fw.write(pdf_bytes)  # 写入数据到文件
            res_form[p]["data"][q]["img_path"] = save_path_i  # 变量赋值
    itemss = extract_huifeng.process_biaotou(ocr_data)  # 变量赋值

    res = {  # 变量赋值
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
    return jsonify(res)  # 返回结果


@app.route("/huifeng_en", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_huifeng_english():  # 定义函数
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    save_dir = request.form.get("save_dir")  # 从表单数据获取参数
    if not os.path.exists(save_dir):  # 检查路径是否存在
        os.makedirs(save_dir)  # 创建目录结构
    b_str = f_obj.read()  # 读取文件内容
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:  # 变量赋值
        temp_file.write(b_str)  # 写入数据到文件
        ocr_data, shape_list = pdf_process_pymu.get_ocr_pymu(temp_file.name)  # 变量赋值
    res_form, iih = extract_huifeng_en.extract_all(ocr_data, shape_list)  # 变量赋值
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)  # 将对象序列化为JSON字符串
    f_obj.seek(0)  # 重置文件指针到开始位置
    b_str_pdf = f_obj.read()  # 读取文件内容
    imgs = pdf2images_fit(b_str_pdf, 4.0)  # 变量赋值
    for p in range(len(res_form)):  # 循环遍历
        for q in range(len(res_form[p]["data"])):  # 循环遍历
            pained_img = extract_meixi_utils.render_one_item(  # 变量赋值
                imgs, res_form, shape_list, iih, p, q, 2
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))  # 拼接文件路径
            img_save = cv2.imencode(".jpg", pained_img)[1]  # 将图像编码为指定格式
            data_encode = np.array(img_save)  # 创建NumPy数组
            str_encode = data_encode.tostring()  # 转换为字节字符串
            pdf_bytes = img2pdf.convert(str_encode)  # 变量赋值
            fw = open(save_path_i, "wb")  # 变量赋值
            fw.write(pdf_bytes)  # 写入数据到文件
            res_form[p]["data"][q]["img_path"] = save_path_i  # 变量赋值
    itemss = extract_huifeng.process_biaotou_en(ocr_data)  # 变量赋值

    res = {  # 变量赋值
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
    return jsonify(res)  # 返回结果


@app.route("/zhaoshang", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_zhaoshang():  # 定义函数
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    save_dir = request.form.get("save_dir")  # 从表单数据获取参数
    if not os.path.exists(save_dir):  # 检查路径是否存在
        os.makedirs(save_dir)  # 创建目录结构
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)  # 变量赋值
    res_form, iih = extract_zhaoshang_utils.extract_all(ocr_data, shape_list)  # 变量赋值
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)  # 将对象序列化为JSON字符串
    f_obj.seek(0)  # 重置文件指针到开始位置
    b_str_pdf = f_obj.read()  # 读取文件内容
    imgs = pdf2images_fit(b_str_pdf, ratio=5.0)  # 变量赋值
    for p in range(len(res_form)):  # 循环遍历
        for q in range(len(res_form[p]["data"])):  # 循环遍历
            pained_img = extract_meixi_utils.render_one_item(  # 变量赋值
                imgs, res_form, shape_list, iih, p, q, 2
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))  # 拼接文件路径
            img_save = cv2.imencode(".jpg", pained_img)[1]  # 将图像编码为指定格式
            data_encode = np.array(img_save)  # 创建NumPy数组
            str_encode = data_encode.tostring()  # 转换为字节字符串
            pdf_bytes = img2pdf.convert(str_encode)  # 变量赋值
            fw = open(save_path_i, "wb")  # 变量赋值
            fw.write(pdf_bytes)  # 写入数据到文件
            res_form[p]["data"][q]["img_path"] = save_path_i  # 变量赋值
    itemss = extract_zhaoshang_utils.process_biaotou(ocr_data)  # 变量赋值
    res = {  # 变量赋值
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
    return jsonify(res)  # 返回结果


@app.route("/taiwan", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_taiwan():  # 定义函数
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    save_dir = request.form.get("save_dir")  # 从表单数据获取参数
    if not os.path.exists(save_dir):  # 检查路径是否存在
        os.makedirs(save_dir)  # 创建目录结构
    # b_str = f_obj.read()
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
    #     temp_file.write(b_str)
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)  # 变量赋值
    res_form, iih = extract_taiwan.extract_all(ocr_data, shape_list)  # 变量赋值
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)  # 将对象序列化为JSON字符串
    f_obj.seek(0)  # 重置文件指针到开始位置
    b_str_pdf = f_obj.read()  # 读取文件内容
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)  # 变量赋值
    for p in range(len(res_form)):  # 循环遍历
        for q in range(len(res_form[p]["data"])):  # 循环遍历
            pained_img = extract_meixi_utils.render_one_item(  # 变量赋值
                imgs, res_form, shape_list, iih, p, q, 2
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))  # 拼接文件路径
            img_save = cv2.imencode(".jpg", pained_img)[1]  # 将图像编码为指定格式
            data_encode = np.array(img_save)  # 创建NumPy数组
            str_encode = data_encode.tostring()  # 转换为字节字符串
            pdf_bytes = img2pdf.convert(str_encode)  # 变量赋值
            fw = open(save_path_i, "wb")  # 变量赋值
            fw.write(pdf_bytes)  # 写入数据到文件
            res_form[p]["data"][q]["img_path"] = save_path_i  # 变量赋值
    itemss = extract_taiwan.process_biaotou(ocr_data)  # 变量赋值
    res = {  # 变量赋值
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
    return jsonify(res)  # 返回结果


@app.route("/huaqi_wuliu", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_huaqi_wuliu():  # 定义函数
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    save_dir = request.form.get("save_dir")  # 从表单数据获取参数
    if not os.path.exists(save_dir):  # 检查路径是否存在
        os.makedirs(save_dir)  # 创建目录结构
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)  # 变量赋值
    ocr_data0 = ocr_data[0]  # 变量赋值
    class_ind = 0  # 变量赋值
    for one_line_text in ocr_data0["text"]:  # 循环遍历
        if "账户名" in one_line_text:  # 条件判断
            class_ind = 1  # 变量赋值
            break

    op_str = "账户对账单查询"  # 变量赋值
    op_cut = 0  # 变量赋值
    for i, text in enumerate(ocr_data0["text"]):  # 循环遍历
        if op_str in text:  # 条件判断
            op_cut = ocr_data0["blob"][i][0]["box"][2][1] + 6  # 变量赋值

    if class_ind == 0:  # 条件判断
        res_form, iih = extract_wuliu_huaqi.extract_all(ocr_data, shape_list)  # 变量赋值
    else:  # 否则执行
        res_form, iih = extract_wuliu_huaqi2.extract_all(ocr_data, shape_list)  # 变量赋值
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)  # 将对象序列化为JSON字符串
    f_obj.seek(0)  # 重置文件指针到开始位置
    b_str_pdf = f_obj.read()  # 读取文件内容
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)  # 变量赋值
    for p in range(len(res_form)):  # 循环遍历
        for q in range(len(res_form[p]["data"])):  # 循环遍历
            pained_img = extract_meixi_utils.render_one_item(  # 变量赋值
                imgs, res_form, shape_list, iih, p, q, 2, op_cut
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))  # 拼接文件路径
            img_save = cv2.imencode(".jpg", pained_img)[1]  # 将图像编码为指定格式
            data_encode = np.array(img_save)  # 创建NumPy数组
            str_encode = data_encode.tostring()  # 转换为字节字符串
            pdf_bytes = img2pdf.convert(str_encode)  # 变量赋值
            fw = open(save_path_i, "wb")  # 变量赋值
            fw.write(pdf_bytes)  # 写入数据到文件
            res_form[p]["data"][q]["img_path"] = save_path_i  # 变量赋值
    res = {  # 变量赋值
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
    return jsonify(res)  # 返回结果


@app.route("/mogeng_ch", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_mogeng_ch():  # 定义函数
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    save_dir = request.form.get("save_dir")  # 从表单数据获取参数
    if not os.path.exists(save_dir):  # 检查路径是否存在
        os.makedirs(save_dir)  # 创建目录结构
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)  # 变量赋值
    ocr_data0 = ocr_data[0]  # 变量赋值
    op_str = "客户参考"  # 变量赋值
    op_cut = 0  # 变量赋值
    for i, text in enumerate(ocr_data0["text"]):  # 循环遍历
        if op_str in text:  # 条件判断
            op_cut = ocr_data0["blob"][i][0]["box"][2][1] - 25  # 变量赋值

    res_form, iih = extract_mogeng.extract_all(ocr_data, shape_list)  # 变量赋值
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)  # 将对象序列化为JSON字符串
    f_obj.seek(0)  # 重置文件指针到开始位置
    b_str_pdf = f_obj.read()  # 读取文件内容
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)  # 变量赋值
    for p in range(len(res_form)):  # 循环遍历
        for q in range(len(res_form[p]["data"])):  # 循环遍历
            pained_img = extract_meixi_utils.render_one_item(  # 变量赋值
                imgs, res_form, shape_list, iih, p, q, 2, op_cut
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))  # 拼接文件路径
            img_save = cv2.imencode(".jpg", pained_img)[1]  # 将图像编码为指定格式
            data_encode = np.array(img_save)  # 创建NumPy数组
            str_encode = data_encode.tostring()  # 转换为字节字符串
            pdf_bytes = img2pdf.convert(str_encode)  # 变量赋值
            fw = open(save_path_i, "wb")  # 变量赋值
            fw.write(pdf_bytes)  # 写入数据到文件
            res_form[p]["data"][q]["img_path"] = save_path_i  # 变量赋值
    itemss = extract_mogeng.process_biaotou_ch(ocr_data0)  # 变量赋值
    res = {  # 变量赋值
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
    return jsonify(res)  # 返回结果


@app.route("/mogeng_en", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_mogeng_en():  # 定义函数
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    save_dir = request.form.get("save_dir")  # 从表单数据获取参数
    if not os.path.exists(save_dir):  # 检查路径是否存在
        os.makedirs(save_dir)  # 创建目录结构
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)  # 变量赋值
    ocr_data0 = ocr_data[0]  # 变量赋值
    op_str = "Value Date Description Customer Ref".replace(" ", "").lower()  # 变量赋值
    op_cut = 0  # 变量赋值
    for i, text in enumerate(ocr_data0["text"]):  # 循环遍历
        if op_str in text:  # 条件判断
            op_cut = ocr_data0["blob"][i][0]["box"][2][1] - 25  # 变量赋值

    res_form, iih = extract_mogeng_en.extract_all(ocr_data, shape_list)  # 变量赋值
    ocr_data_ss = json.dumps(ocr_data, cls=NumpyEncoder)  # 将对象序列化为JSON字符串
    f_obj.seek(0)  # 重置文件指针到开始位置
    b_str_pdf = f_obj.read()  # 读取文件内容
    imgs = pdf2images_fit(b_str_pdf, ratio=4.0)  # 变量赋值
    for p in range(len(res_form)):  # 循环遍历
        for q in range(len(res_form[p]["data"])):  # 循环遍历
            pained_img = extract_meixi_utils.render_one_item(  # 变量赋值
                imgs, res_form, shape_list, iih, p, q, 2, op_cut
            )
            save_path_i = os.path.join(save_dir, "{}_{}.pdf".format(p, q))  # 拼接文件路径
            img_save = cv2.imencode(".jpg", pained_img)[1]  # 将图像编码为指定格式
            data_encode = np.array(img_save)  # 创建NumPy数组
            str_encode = data_encode.tostring()  # 转换为字节字符串
            pdf_bytes = img2pdf.convert(str_encode)  # 变量赋值
            fw = open(save_path_i, "wb")  # 变量赋值
            fw.write(pdf_bytes)  # 写入数据到文件
            res_form[p]["data"][q]["img_path"] = save_path_i  # 变量赋值
    itemss = extract_mogeng.process_biaotou_en(ocr_data0)  # 变量赋值
    res = {  # 变量赋值
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
    return jsonify(res)  # 返回结果


def fix_min_len_resize(img, min_l, max_len=2500):  # 定义函数
    h, w = img.shape[0:2]  # 变量赋值
    ratio = float(min_l) / min(h, w)  # 变量赋值
    ratio = min(max_len / max(h, w), ratio)  # 变量赋值
    new_h, new_w = int(ratio * h), int(ratio * w)  # 变量赋值
    res_img = cv2.resize(img, (new_w, new_h))  # 变量赋值
    res_ratio = (float(w) / new_w, float(h) / new_h)  # 变量赋值
    return res_img, res_ratio  # 返回结果


def min_max_h_span(box_list):  # 定义函数
    min_h = 10000  # 变量赋值
    max_h = 0  # 变量赋值
    for box_i in box_list:  # 循环遍历
        max_i = np.max(box_i[:, 1])  # 变量赋值
        min_i = np.min(box_i[:, 1])  # 变量赋值
        if max_i > max_h:  # 条件判断
            max_h = max_i  # 变量赋值

        if min_i < min_h:  # 条件判断
            min_h = min_i  # 变量赋值

    return (min_h, max_h)  # 返回结果


@app.route("/huifeng_yinni", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_huifeng_yinni():  # 定义函数
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    save_dir = request.form.get("save_dir")  # 从表单数据获取参数
    if not os.path.exists(save_dir):  # 检查路径是否存在
        os.makedirs(save_dir)  # 创建目录结构
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)  # 变量赋值
    f_obj.seek(0)  # 重置文件指针到开始位置
    b_str = f_obj.read()  # 读取文件内容
    image_list = pdf2images_fit(b_str)  # 变量赋值
    image_list = [fix_min_len_resize(img, 1600)[0] for img in image_list]  # 变量赋值
    res = []  # 变量赋值
    for p, img in enumerate(image_list):  # 循环遍历
        tmp_pa = {"page": p, "data": [], "regin": [], "shape_list": (2264, 1600)}  # 变量赋值
        contour_analyser_list = table_model.table_predict(img, ocr_data[p])  # 变量赋值
        for l, contour_analyser_i in enumerate(contour_analyser_list):  # 循环遍历
            df = pd.read_html(StringIO(contour_analyser_i.html))[0]  # 变量赋值
            # 清理数据：替换换行符并去除多余空格
            df = df.applymap(  # 变量赋值
                lambda x: x.replace("\n", " ") if isinstance(x, str) else x
            )
            df = df.applymap(  # 变量赋值
                lambda x: " ".join(str(x).split()) if isinstance(x, str) else x
            )
            df = df.fillna("")  # 变量赋值
            pos2ind_map = {}  # 变量赋值
            for i, pot in enumerate(contour_analyser_i.cell_table_coord):  # 循环遍历

                pos2ind_map[tuple(list(pot))] = i  # 变量赋值

            # 替换空字符串为 NaN
            # df.replace('', np.nan, inplace=True)

            # 转换为 NumPy 数组
            numpy_table = df.to_numpy().astype(str)  # 变量赋值
            top_region = []  # 变量赋值
            for m in range(numpy_table.shape[1]):  # 循环遍历

                ind_p = pos2ind_map[(m, 0)]  # 变量赋值

                box_p = contour_analyser_i.cell_hull_coord[ind_p]  # 变量赋值
                top_region.append(box_p)  # 添加元素到列表
            biaotou_span = min_max_h_span(top_region)  # 变量赋值

            img_cut_biaotou = img[0 : int(biaotou_span[1]), :, :]  # 变量赋值
            for i in range(1, numpy_table.shape[0]):  # 循环遍历
                psa = {}  # 变量赋值
                region_box_line_i = []  # 变量赋值

                cla = numpy_table[0][0].replace(" ", "")  # 变量赋值
                if cla in ["AccountLocation", "Accountcurrency"]:  # 条件判断
                    clb = numpy_table[i][0].replace(" ", "")  # 变量赋值
                    if clb in ["PHTotal:", "SGTotal:", "THTotal:", "MYTotal:"]:  # 条件判断
                        break

                cla = numpy_table[0][1].replace(" ", "")  # 变量赋值
                if cla in ["AccountLocation", "Accountcurrency"]:  # 条件判断
                    clb = numpy_table[i][1].replace(" ", "")  # 变量赋值
                    if clb in ["PHTotal:", "SGTotal:", "THTotal:", "MYTotal:"]:  # 条件判断
                        break

                for j in range(0, numpy_table.shape[1]):  # 循环遍历

                    try:
                        ind_ij = pos2ind_map[(j, i)]  # 变量赋值
                        if j == 0:  # 条件判断
                            pass
                        else:  # 否则执行
                            region_box_line_i.append(  # 添加元素到列表
                                contour_analyser_i.cell_hull_coord[ind_ij]
                            )

                        psa[numpy_table[0][j].replace(" ", "")] = numpy_table[i][j]  # 变量赋值
                    except:
                        psa[numpy_table[0][j].replace(" ", "")] = ""  # 变量赋值

                ele_span_i = min_max_h_span(region_box_line_i)  # 变量赋值
                img_cut_i = img[int(ele_span_i[0]) : int(ele_span_i[1]), :, :]  # 变量赋值
                img_merge = np.concatenate([img_cut_biaotou, img_cut_i], axis=0)  # 变量赋值
                img_save = cv2.imencode(".jpg", img_merge)[1]  # 将图像编码为指定格式
                data_encode = np.array(img_save)  # 创建NumPy数组
                str_encode = data_encode.tostring()  # 转换为字节字符串
                pdf_bytes = img2pdf.convert(str_encode)  # 变量赋值
                psa["img_path"] = os.path.join(save_dir, "{}_{}_{}.pdf".format(p, l, i))  # 拼接文件路径
                fw = open(psa["img_path"], "wb")  # 变量赋值
                fw.write(pdf_bytes)  # 写入数据到文件
                tmp_pa["data"].append(psa)  # 添加元素到列表
        res.append(tmp_pa)  # 添加元素到列表
    all_res = {  # 变量赋值
        "data": {"form": res, "item": {}, "shape_list": [], "ocr_data": ""},
        "code": 200,
        "message": "ok",
    }
    return jsonify(all_res)  # 返回结果


@app.route("/is_pdf_parse", methods=["POST", "GET"])  # 注册API路由
def is_pdf_parse():  # 定义函数
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)  # 变量赋值
    ocr_data0 = ocr_data[0]  # 变量赋值
    has_ind_num = 0  # 变量赋值
    for line_text in ocr_data0["text"]:  # 循环遍历
        if len(line_text) > 0:  # 条件判断
            has_ind_num += 1  # 变量赋值
    if has_ind_num >= 3:  # 条件判断
        res = {"is_pdf_parse": 1, "code": 200, "message": "ok"}  # 变量赋值
    else:  # 否则执行
        res = {"is_pdf_parse": 0, "code": 200, "message": "ok"}  # 变量赋值
    return jsonify(res)  # 返回结果


@app.route("/render_file", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_file():  # 定义函数
    f_json = request.get_json()  # 变量赋值
    path = f_json["file"]  # 变量赋值
    res_blob = f_json["form"]  # 变量赋值
    thickness = f_json["thickness"]  # 变量赋值
    page = f_json["page"]  # 变量赋值
    line_ind = f_json["line_ind"]  # 变量赋值
    shape_list = f_json["shape_list"]  # 变量赋值
    iis = f_json["head_h"]  # 变量赋值
    f = open(path, "rb")  # 变量赋值
    b_str_pdf = f.read()  # 读取文件内容
    imgs = pdf2images_fit(b_str_pdf)  # 变量赋值
    final_img = extract_meixi_utils.render_one_item(  # 变量赋值
        imgs, res_blob, shape_list, iis, page, line_ind, thickness
    )
    img_save = cv2.imencode(".jpg", final_img)[1]  # 将图像编码为指定格式
    data_encode = np.array(img_save)  # 创建NumPy数组
    str_encode = data_encode.tostring()  # 转换为字节字符串
    b64_str_encode_render = base64.b64encode(str_encode).decode("utf-8")  # 对字节数据进行Base64编码
    res = {"message": "ok", "code": 200, "data": b64_str_encode_render}  # 变量赋值
    return jsonify(res)  # 返回结果


@app.route("/render", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_rrr():  # 定义函数
    f_json = request.get_json()  # 变量赋值
    bs64_pdf_str = f_json["file"]  # 变量赋值
    res_blob = f_json["form"]  # 变量赋值
    thickness = f_json["thickness"]  # 变量赋值
    page = f_json["page"]  # 变量赋值
    line_ind = f_json["line_ind"]  # 变量赋值
    shape_list = f_json["shape_list"]  # 变量赋值
    iis = f_json["head_h"]  # 变量赋值
    b_str_pdf_base64 = bs64_pdf_str.encode("utf-8")  # 变量赋值
    b_str_pdf = base64.b64decode(b_str_pdf_base64)  # 对Base64字符串进行解码
    imgs = pdf2images_fit(b_str_pdf)  # 变量赋值
    final_img = extract_meixi_utils.render_one_item(  # 变量赋值
        imgs, res_blob, shape_list, iis, page, line_ind, thickness
    )
    img_save = cv2.imencode(".jpg", final_img)[1]  # 将图像编码为指定格式
    data_encode = np.array(img_save)  # 创建NumPy数组
    str_encode = data_encode.tostring()  # 转换为字节字符串
    b64_str_encode_render = base64.b64encode(str_encode).decode("utf-8")  # 对字节数据进行Base64编码
    res = {"message": "ok", "code": 200, "data": b64_str_encode_render}  # 变量赋值
    return jsonify(res)  # 返回结果


@app.route("/render_touming", methods=["POST", "GET"])  # 注册API路由
def post_Data_Cetifas_rrr_touming():  # 定义函数
    f_json = request.get_json()  # 变量赋值
    bs64_pdf_str = f_json["file"]  # 变量赋值
    res_blob = f_json["form"]  # 变量赋值
    thickness = f_json["thickness"]  # 变量赋值
    page = f_json["page"]  # 变量赋值
    line_ind = f_json["line_ind"]  # 变量赋值
    shape_list = f_json["shape_list"]  # 变量赋值
    iis = f_json["head_h"]  # 变量赋值
    b_str_pdf_base64 = bs64_pdf_str.encode("utf-8")  # 变量赋值
    b_str_pdf = base64.b64decode(b_str_pdf_base64)  # 对Base64字符串进行解码
    imgs = pdf2images_fit(b_str_pdf)  # 变量赋值
    final_img = extract_meixi_utils.render_one_item(  # 变量赋值
        imgs, res_blob, shape_list, iis, page, line_ind, thickness
    )
    img_save = cv2.imencode(".jpg", final_img)[1]  # 将图像编码为指定格式
    data_encode = np.array(img_save)  # 创建NumPy数组
    str_encode = data_encode.tostring()  # 转换为字节字符串
    b64_str_encode_render = base64.b64encode(str_encode).decode("utf-8")  # 对字节数据进行Base64编码
    res = {"message": "ok", "code": 200, "data": b64_str_encode_render}  # 变量赋值
    return jsonify(res)  # 返回结果


@app.route("/classify_pdf", methods=["POST", "GET"])  # 注册API路由
def classify_pdf():  # 定义函数
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    config_str = request.form.get("config")  # 从表单数据获取参数
    config_json = json.loads(config_str)  # 将JSON字符串解析为对象
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)  # 变量赋值
    class_name = classify_utils.classify_run(ocr_data, config_json)  # 变量赋值
    res = {"message": "ok", "code": 200, "data": class_name}  # 变量赋值
    return jsonify(res)  # 返回结果


@app.route("/mogeng_merge", methods=["POST", "GET"])  # 注册API路由
def mogeng_merge():  # 定义函数

    url = "http://127.0.0.1:{}/classify_pdf".format(port)  # 变量赋值
    payload = {  # 变量赋值
        "config": '{"page":0,"data":{"en":{"da":["Description Customer Ref"],"no":["客户参考 银行参考"]},"ch":{"da":["客户参考 银行参考"],"no":["Description Customer Ref"]}}}'
    }
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    save_dir = request.form.get("save_dir")  # 从表单数据获取参数
    files = [("file", (f_obj.name, f_obj, "application/pdf"))]  # 变量赋值
    headers = {}  # 变量赋值
    response = requests.request("POST", url, headers=headers, data=payload, files=files)  # 变量赋值
    f_obj.seek(0)  # 重置文件指针到开始位置
    cc = response.text  # 变量赋值
    cc = json.loads(cc)  # 将JSON字符串解析为对象

    if cc["data"] == "ch":  # 条件判断
        url = "http://127.0.0.1:{}/mogeng_ch".format(port)  # 变量赋值
        payload = {"save_dir": save_dir}  # 变量赋值
        files = [("file", (f_obj.name, f_obj, "application/pdf"))]  # 变量赋值
        headers = {}  # 变量赋值
        response = requests.request(  # 变量赋值
            "POST", url, headers=headers, data=payload, files=files  # 变量赋值
        )

    else:  # 否则执行
        url = "http://127.0.0.1:{}/mogeng_en".format(port)  # 变量赋值
        payload = {"save_dir": save_dir}  # 变量赋值
        files = [("file", (f_obj.name, f_obj, "application/pdf"))]  # 变量赋值
        headers = {}  # 变量赋值
        response = requests.request(  # 变量赋值
            "POST", url, headers=headers, data=payload, files=files  # 变量赋值
        )
    cc = response.text  # 变量赋值
    cc = json.loads(cc)  # 将JSON字符串解析为对象
    return jsonify(cc)  # 返回结果


@app.route("/huifeng_merge", methods=["POST", "GET"])  # 注册API路由
def huifeng_merge():  # 定义函数

    url = "http://127.0.0.1:{}/classify_pdf".format(port)  # 变量赋值
    payload = {  # 变量赋值
        "config": '{"page":0,"data":{"en":{"da":["Bank reference"],"no":["客户参考"]},"ch":{"da":["客户参考"],"no":["Bank reference"]}}}'
    }
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    save_dir = request.form.get("save_dir")  # 从表单数据获取参数
    files = [("file", (f_obj.name, f_obj, "application/pdf"))]  # 变量赋值
    headers = {}  # 变量赋值
    response = requests.request("POST", url, headers=headers, data=payload, files=files)  # 变量赋值
    f_obj.seek(0)  # 重置文件指针到开始位置
    cc = response.text  # 变量赋值
    cc = json.loads(cc)  # 将JSON字符串解析为对象

    if cc["data"] == "ch":  # 条件判断
        url = "http://127.0.0.1:{}/huifeng_ch".format(port)  # 变量赋值
        payload = {"save_dir": save_dir}  # 变量赋值
        files = [("file", (f_obj.name, f_obj, "application/pdf"))]  # 变量赋值
        headers = {}  # 变量赋值
        response = requests.request(  # 变量赋值
            "POST", url, headers=headers, data=payload, files=files  # 变量赋值
        )

    else:  # 否则执行
        url = "http://127.0.0.1:{}/huifeng_en".format(port)  # 变量赋值
        payload = {"save_dir": save_dir}  # 变量赋值
        files = [("file", (f_obj.name, f_obj, "application/pdf"))]  # 变量赋值
        headers = {}  # 变量赋值
        response = requests.request(  # 变量赋值
            "POST", url, headers=headers, data=payload, files=files  # 变量赋值
        )
    cc = response.text  # 变量赋值
    cc = json.loads(cc)  # 将JSON字符串解析为对象
    return jsonify(cc)  # 返回结果


@app.route("/huifeng_cut", methods=["POST", "GET"])  # 注册API路由
def huifeng_cut():  # 定义函数
    biyou = "AdviceType通知种类"  # 变量赋值
    f_obj_tt = request.files  # 获取请求中的上传文件
    f_obj = f_obj_tt.get("file")  # 变量赋值
    ocr_data, shape_list = pdf_process.get_ocr(f_obj)  # 变量赋值
    f_obj.seek(0)  # 重置文件指针到开始位置
    biyou_id_list = []  # 变量赋值
    for i, ocr_data_i in enumerate(ocr_data):  # 循环遍历
        text_all_i = "".join(ocr_data_i["raw_text"])  # 变量赋值
        text_all_i = text_all_i.replace(" ", "")  # 变量赋值
        if biyou in text_all_i:  # 条件判断
            biyou_id_list.append(i)  # 添加元素到列表
    res_span = []  # 变量赋值

    for i in range(len(biyou_id_list)):  # 循环遍历
        if i + 1 < len(biyou_id_list):  # 条件判断
            res_span.append((biyou_id_list[i], biyou_id_list[i + 1]))  # 添加元素到列表
        else:  # 否则执行
            res_span.append((biyou_id_list[i], len(ocr_data)))  # 添加元素到列表

    res = []  # 变量赋值

    pdf_reader = PdfReader(f_obj)  # 变量赋值
    for i, span_i in enumerate(res_span):  # 循环遍历
        pdf_writer = PdfWriter()  # 变量赋值
        for j in range(span_i[0], span_i[1]):  # 循环遍历
            pdf_writer.add_page(pdf_reader.pages[j])
        output_buffer = BytesIO()  # 变量赋值
        pdf_writer.write(output_buffer)  # 写入数据到文件
        pdf_bytes = output_buffer.getvalue()  # 变量赋值
        output_buffer.close()  # 关闭文件句柄释放资源
        b64_str_encode_render = base64.b64encode(pdf_bytes).decode("utf-8")  # 对字节数据进行Base64编码
        res.append(b64_str_encode_render)  # 添加元素到列表
    # print('aaa')

    return {"message": "ok", "code": 200, "data": res}  # 返回结果


if __name__ == "__main__":  # 条件判断
    app.run(host="0.0.0.0", port=port, threaded=True)  # 变量赋值
    # ttt = pywsgi.WSGIServer.get_environ()
    # server = pywsgi.WSGIServer(('0.0.0.0', port), app)
    # ttt = server.get_environ()
    # server.serve_forever()
