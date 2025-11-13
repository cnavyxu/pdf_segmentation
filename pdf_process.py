import cv2  # 导入OpenCV计算机视觉库
import numpy as np  # 导入NumPy库用于数值计算和数组操作
import pdfplumber
import fitz  # 导入PyMuPDF库用于PDF处理
from collections import defaultdict  # 导入默认字典类
import copy
import time  # 导入时间处理模块
import datetime


def two_to_four_convert(points, ratio):  # 定义函数
    res = np.zeros([4, 2])  # 变量赋值
    res[0][0] = points[0] * ratio  # 变量赋值
    res[0][1] = points[1] * ratio  # 变量赋值
    res[1][0] = points[2] * ratio  # 变量赋值
    res[1][1] = points[1] * ratio  # 变量赋值
    res[2][0] = points[2] * ratio  # 变量赋值
    res[2][1] = points[3] * ratio  # 变量赋值
    res[3][0] = points[0] * ratio  # 变量赋值
    res[3][1] = points[3] * ratio  # 变量赋值
    return res  # 返回结果


def  w_cluster_has_blank(cluster_obj, r_shape, ratio, crop_box=None):  # 定义函数
    r_w, r_h = r_shape  # 变量赋值
    res = []  # 变量赋值
    for i in range(len(cluster_obj)):  # 循环遍历
        resi = []  # 变量赋值
        obj_line = cluster_obj[i]  # 变量赋值
        if len(obj_line) == 0:  # 条件判断
            continue
        tmp_list = []  # 变量赋值
        tmp_text = ''  # 变量赋值
        span = []  # 变量赋值
        ts = sorted(obj_line, key=lambda a: a['x0'])  # 变量赋值
        # ts = obj_line
        ts = move_overlap(ts)  # 变量赋值
        for obj in ts:  # 循环遍历

            if not crop_box is None:  # 条件判断
                if obj['x0'] <= crop_box[0] or obj['x0'] >= crop_box[2] or obj['y0'] <= crop_box[1] or obj['y1'] >= \
                        crop_box[3]:
                    continue

                obj['x0'] -= crop_box[0]  # 变量赋值
                obj['x1'] -= crop_box[0]  # 变量赋值
                obj['y0'] -= crop_box[1]  # 变量赋值
                obj['y1'] -= crop_box[1]  # 变量赋值

            if tmp_list == []:  # 条件判断
                if obj['text'] == '':  # 条件判断
                    continue
                tmp_list.append((obj['x0'], obj['y0'], obj['x1'], obj['y1']))  # 添加元素到列表
                tmp_text += obj['text']  # 变量赋值
                span.append(((obj['x0'] + obj['x1']) / 2 * ratio, (r_h - (obj['y0'] + obj['y1']) / 2) * ratio))  # 添加元素到列表
            else:  # 否则执行
                # assert obj['x0']+0.5 >= tmp_list[-1][2]
                bb = obj['x0'] - tmp_list[-1][2]  # 变量赋值
                if abs(bb) <= 2:  # 条件判断
                    # if obj['x0'] - tmp_list[-1][2] <= 6:
                    if obj['text'] == '':  # 条件判断
                        continue
                    tmp_list.append((obj['x0'], obj['y0'], obj['x1'], obj['y1']))  # 添加元素到列表
                    tmp_text += obj['text']  # 变量赋值
                    span.append(((obj['x0'] + obj['x1']) / 2 * ratio, (r_h - (obj['y0'] + obj['y1']) / 2) * ratio))  # 添加元素到列表
                else:  # 否则执行
                    tmp_blob = {  # 变量赋值
                        # 'box': [tmp_list[0][0], r_h - tmp_list[0][1], tmp_list[-1][2], r_h - tmp_list[-1][3]],
                        'box': two_to_four_convert(
                            [tmp_list[0][0], r_h - tmp_list[-1][3], tmp_list[-1][2], r_h - tmp_list[0][1]], ratio),
                        'text': tmp_text,
                        'span': span

                    }
                    resi.append(tmp_blob)  # 添加元素到列表
                    tmp_list = []  # 变量赋值
                    tmp_text = ''  # 变量赋值
                    span = []  # 变量赋值
                    if obj['text'] == '':  # 条件判断
                        continue
                    tmp_list.append((obj['x0'], obj['y0'], obj['x1'], obj['y1']))  # 添加元素到列表
                    tmp_text += obj['text']  # 变量赋值
                    span.append(((obj['x0'] + obj['x1']) / 2 * ratio, (r_h - (obj['y0'] + obj['y1']) / 2) * ratio))  # 添加元素到列表
        if not tmp_list == []:  # 条件判断
            tmp_blob = {  # 变量赋值
                # 'box': [tmp_list[0][0], r_h - tmp_list[0][1], tmp_list[-1][2], r_h - tmp_list[-1][3]],
                'box': two_to_four_convert(
                    [tmp_list[0][0], r_h - tmp_list[-1][3], tmp_list[-1][2], r_h - tmp_list[0][1]], ratio),
                'text': tmp_text,
                'span': span
            }
            if tmp_text == '':  # 条件判断
                pass
            else:  # 否则执行
                resi.append(tmp_blob)  # 添加元素到列表
        res.append(resi)  # 添加元素到列表
    return res  # 返回结果


def move_overlap(ts_line):  # 定义函数
    # 删除一行加粗的重叠字符
    last_span = (0, 1)  # 变量赋值
    res = []  # 变量赋值
    for ts in ts_line:  # 循环遍历
        this_span = (ts['x0'], ts['x1'])  # 变量赋值
        if (this_span[1] - this_span[0]) < 0.01:  # 条件判断
            continue
        iou = (min(this_span[1], last_span[1]) - max(this_span[0], last_span[0])) / (this_span[1] - this_span[0])  # 变量赋值
        if iou > 0.7:  # 条件判断
            continue
        else:  # 否则执行
            last_span = this_span  # 变量赋值
            res.append(ts)  # 添加元素到列表
    return res  # 返回结果


def w_cluster(cluster_obj, r_shape, ratio, page_box):  # 定义函数
    r_w, r_h = r_shape  # 变量赋值
    res = []  # 变量赋值
    for i in range(len(cluster_obj)):  # 循环遍历
        resi = []  # 变量赋值
        obj_line = cluster_obj[i]  # 变量赋值
        if len(obj_line) == 0:  # 条件判断
            continue
        tmp_list = []  # 变量赋值
        tmp_text = ''  # 变量赋值
        span = []  # 变量赋值
        ts = sorted(obj_line, key=lambda a: a['x0'])  # 变量赋值
        ts = move_overlap(ts)  # 变量赋值
        for obj in ts:  # 循环遍历
            if obj['x0'] < -10 + page_box[0] or obj['y0'] < -10+page_box[1] or obj['x1'] > page_box[2] + 10 or obj['y1'] > page_box[3] + 10:  # 条件判断
                # print('aaa')
                continue
            if tmp_list == []:  # 条件判断
                if obj['text'] == '' or obj['text'] == ' ':  # 条件判断
                    continue
                tmp_list.append((obj['x0'], obj['y0'], obj['x1'], obj['y1']))  # 添加元素到列表
                tmp_text += obj['text']  # 变量赋值
                span.append(((obj['x0'] + obj['x1']) / 2 * ratio, (r_h - (obj['y0'] + obj['y1']) / 2) * ratio))  # 添加元素到列表
                continue
            else:  # 否则执行
                # assert obj['x0']+0.5 >= tmp_list[-1][2]
                if obj['x0'] - tmp_list[-1][2] <= 8:  # 条件判断
                    if obj['text'] == '' or obj['text'] == ' ':  # 条件判断
                        continue
                    tmp_list.append((obj['x0'], obj['y0'], obj['x1'], obj['y1']))  # 添加元素到列表
                    tmp_text += obj['text']  # 变量赋值
                    span.append(((obj['x0'] + obj['x1']) / 2 * ratio, (r_h - (obj['y0'] + obj['y1']) / 2) * ratio))  # 添加元素到列表
                else:  # 否则执行
                    tmp_blob = {  # 变量赋值
                        # 'box': [tmp_list[0][0], r_h - tmp_list[0][1], tmp_list[-1][2], r_h - tmp_list[-1][3]],
                        'box': two_to_four_convert(
                            [tmp_list[0][0], r_h - tmp_list[-1][3], tmp_list[-1][2], r_h - tmp_list[0][1]], ratio),
                        'text': tmp_text,
                        'span': span

                    }
                    resi.append(tmp_blob)  # 添加元素到列表
                    tmp_list = []  # 变量赋值
                    tmp_text = ''  # 变量赋值
                    span = []  # 变量赋值
                    if obj['text'] == '' or obj['text'] == ' ':  # 条件判断
                        continue
                    tmp_list.append((obj['x0'], obj['y0'], obj['x1'], obj['y1']))  # 添加元素到列表
                    tmp_text += obj['text']  # 变量赋值
                    span.append(((obj['x0'] + obj['x1']) / 2 * ratio, (r_h - (obj['y0'] + obj['y1']) / 2) * ratio))  # 添加元素到列表
        if not tmp_list == []:  # 条件判断
            tmp_blob = {  # 变量赋值
                # 'box': [tmp_list[0][0], r_h - tmp_list[0][1], tmp_list[-1][2], r_h - tmp_list[-1][3]],
                'box': two_to_four_convert(
                    [tmp_list[0][0], r_h - tmp_list[-1][3], tmp_list[-1][2], r_h - tmp_list[0][1]], ratio),
                'text': tmp_text,
                'span': span
            }
            resi.append(tmp_blob)  # 添加元素到列表
        res.append(resi)  # 添加元素到列表
    return res  # 返回结果


def get_r_ratio(image_info):  # 定义函数
    max_ratio = 1  # 变量赋值
    for image_info_i in image_info:  # 循环遍历
        tmp_h, tmp_w = image_info_i['bbox'][3] - image_info_i['bbox'][1], image_info_i['bbox'][2] - \
                       image_info_i['bbox'][0]
        g_h, g_w = image_info_i["height"], image_info_i["width"]  # 变量赋值
        ratio = max(g_w / tmp_w, g_h / tmp_h)  # 变量赋值
        if ratio > max_ratio and ratio <= 5:  # 条件判断
            max_ratio = ratio  # 变量赋值
    return max_ratio  # 返回结果


def get_rotate_angle(der_matrix):  # 定义函数
    res = 0  # 变量赋值
    dist = 100  # 变量赋值
    rot_list = [0, 90, 180, 270]  # 变量赋值
    for rot in rot_list:  # 循环遍历
        rot_matrix = fitz.Matrix(rot)  # 变量赋值
        dist_i = abs(der_matrix[0] - rot_matrix[0]) + abs(der_matrix[1] - rot_matrix[1]) + abs(  # 变量赋值
            der_matrix[2] - rot_matrix[2]) + abs(der_matrix[3] - rot_matrix[3])
        if dist_i < dist:  # 条件判断
            dist = dist_i  # 变量赋值
            res = rot  # 变量赋值
    return res  # 返回结果


def ReadPdfImage(filepath=None, savepath=None, **kwargs):  # 定义函数
    '''
    读取pdf中的图片，返回图片数组，格式为pil 图片类型
    :param filepath:
    :param pages:
    :param kwargs:
    :return:
    '''
    doc = fitz.open(filepath)  # 打开文件或资源
    images = []  # 变量赋值
    for i, page in enumerate(doc):  # 循环遍历
        canvos_point1 = (round(page.CropBox.top_left[0]), round(page.CropBox.top_left[1]))  # 变量赋值
        canvos_point2 = (round(page.CropBox.bottom_right[0]), round(page.CropBox.bottom_right[1]))  # 变量赋值
        canvos_w, canvos_h = canvos_point2[0] - canvos_point1[0], canvos_point2[1] - canvos_point1[1]  # 变量赋值
        canvos = np.ones([canvos_h, canvos_w, 3]) * 255  # 变量赋值
        image_info = page.get_image_info(xrefs=True)  # 变量赋值

        # shrink = fitz.Matrix(1 / 2340, 0, 0, 1 / 1654, 0, 0)
        # imgrect = fitz.Rect(0, 0, 2340, 1654)
        # bbox_dd, transform_dd = page.get_image_bbox('Im1', transform=True)
        # cc = shrink * transform_dd
        image_list = page.get_images()  # 变量赋值
        xref_list = {st['xref']: st for st in image_info}  # 变量赋值
        ratio = get_r_ratio(image_info)  # 变量赋值
        canvos = cv2.resize(canvos, None, None, ratio, ratio)  # 变量赋值
        s_list = []  # 变量赋值

        img_name_num_map = defaultdict(int)  # 变量赋值
        for im_info_i in image_list:  # 循环遍历
            name_i = im_info_i[7]  # 变量赋值
            img_name_num_map[name_i] += 1  # 变量赋值

        for c, item in enumerate(image_list):  # 循环遍历
            try:
                xref = item[0]  # 变量赋值
                if not xref in xref_list:  # 条件判断
                    continue
                item_name = item[7]  # 变量赋值
                if img_name_num_map[item_name] >= 2:  # 条件判断
                    continue
                # todo 这边可能重名， 从而影响到
                box, transform_ed = page.get_image_bbox(item_name, transform=True)  # 变量赋值
                point1 = (round(box.top_left.x), round(box.top_left.y))  # 变量赋值
                point2 = (round(box.bottom_right.x), round(box.bottom_right.y))  # 变量赋值
                pix = fitz.Pixmap(doc, xref)  # 变量赋值
                # ddf = pix.colorspace
                # ccs = pix.is_monochrome
                colorspace = xref_list[xref]['colorspace']  # 变量赋值

                tt = pix.tobytes()  # 转换为字节串
                # nparr = np.fromstring(img['image'], np.uint8)
                nparr = np.fromstring(tt, np.uint8)  # 变量赋值
                img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 将字节流解码为图像
                if colorspace == 0:  # 条件判断
                    img_decode = 255 - img_decode  # 变量赋值
                # cv2.imwrite('/mnt/e/data/ocr/保单/visual2/{}_{}_aa.png'.format(i, c), img_decode)
                transform = xref_list[xref]['transform']  # 变量赋值
                if transform[0] < 0:  # 条件判断
                    img_decode = np.flip(img_decode, axis=1)  # 变量赋值
                if transform[3] < 0:  # 条件判断
                    img_decode = np.flip(img_decode, axis=0)  # 变量赋值

                if point1[0] < -1 or point1[1] < -1 or point2[0] < -1 or point2[1] < -1:  # 条件判断
                    new_point2 = (point2[0] - point1[0], point2[1] - point1[1])  # 变量赋值
                    new_point1 = (0, 0)  # 变量赋值
                    point1 = new_point1  # 变量赋值
                    point2 = new_point2  # 变量赋值

                ratio_s = img_decode.shape[1] / (point2[0] - point1[0])  # todo 这里用w的ratio更合理
                s_list.append((img_decode.shape[0] / (point2[1] - point1[1]),  # 添加元素到列表
                               img_decode.shape[1] / (point2[0] - point1[0]), point1, point2, img_decode.shape))
                d_ratio = ratio / ratio_s  # 变量赋值
                img_decode = cv2.resize(img_decode, None, None, d_ratio, d_ratio, cv2.INTER_CUBIC)  # 变量赋值
                point1 = (round(point1[0] * ratio), round(point1[1] * ratio))  # 变量赋值
                point2 = (round(point2[0] * ratio), round(point2[1] * ratio))  # 变量赋值
                # img_decode =  cv2.resize(img_decode, (point2[0]-point1[0], point2[1]-point1[1]))
                # max_h = min(canvos.shape[0]-point1[1], point2[1]-point1[1], img_decode.shape[0])
                # max_w = min(canvos.shape[1]- point1[0], point2[0]- point1[0], img_decode.shape[1])
                max_h = min(canvos.shape[0] - point1[1], img_decode.shape[0])  # 变量赋值
                max_w = min(canvos.shape[1] - point1[0], img_decode.shape[1])  # 变量赋值
                canvos[point1[1]:point1[1] + max_h, point1[0]:point1[0] + max_w, :] = img_decode[0:max_h, 0:max_w, :]  # 变量赋值
            except:
                continue
        rotate_angle = get_rotate_angle(page.derotation_matrix)  # 变量赋值
        r_canvos = np.rot90(canvos, rotate_angle // 90)  # 变量赋值
        images.append(r_canvos)  # 添加元素到列表
    return images  # 返回结果


def wrap_sorted_line_pdf(sorted_line):  # 定义函数
    res = {}  # 变量赋值
    res['span'] = []  # 变量赋值
    res['text'] = []  # 变量赋值
    res['ind'] = []  # 变量赋值
    res['blob'] = []  # 变量赋值
    for s, single_line in enumerate(sorted_line):  # 循环遍历
        line_text = ""  # 变量赋值
        position_list = []  # 变量赋值
        tmp_ind_list = []  # 变量赋值
        res_single_line = []  # 变量赋值
        ind_sum = 0  # 变量赋值
        for blob in single_line:  # 循环遍历
            # def blob_box_wrap(box):
            #     return np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
            # blob['box'] = blob_box_wrap(blob['box'])
            blob['text'] = blob['text'].replace('（', '(').replace('）', ')').replace('，', ',')  # 变量赋值
            if blob['text'] == '':  # 条件判断
                continue
            kk = {  # 变量赋值
                'box': blob['box'],
                'text': blob['text']
            }
            line_text += blob['text']  # 变量赋值
            # tmp = Span2SinTextPoint_raw(blob['box'], blob['span'])
            position_list.extend(blob['span'])  # 扩展列表
            ind_sum += len(blob['text'])  # 变量赋值
            tmp_ind_list.append(ind_sum)  # 添加元素到列表
            res_single_line.append(kk)  # 添加元素到列表
        if line_text.replace(' ', '') == '':  # 条件判断
            continue
        res['span'].append(position_list)  # 添加元素到列表
        res['text'].append(line_text)  # 添加元素到列表
        res['ind'].append(tmp_ind_list)  # 添加元素到列表
        res['blob'].append(res_single_line)  # 添加元素到列表
        res['raw_text'] = copy.deepcopy(res['text'])  # 变量赋值
    return res  # 返回结果


def Span2SinTextPoint_raw(cnt, span):  # 定义函数
    h = cnt[3][1] - cnt[0][1]  # 变量赋值
    h_start = (cnt[0][1] + cnt[3][1]) / 2  # 变量赋值
    w_start = cnt[0][0]  # 变量赋值
    w_l = cnt[1][0] - cnt[0][0]  # 变量赋值
    h_off_set = cnt[1][1] - cnt[0][1]  # 变量赋值
    # tan_theta = abs((cnt[1][1]-cnt[1][0])/(cnt[1][0] - cnt[0][0]))

    w_list = [int(w_start + (h / 32.0) * 8 * p - h / 32.0 * 30) for p in span]  # 变量赋值

    h_list = [int(h_start + h_off_set * (p / w_l)) for p in span]  # 变量赋值

    res = [(w_list[i], h_list[i]) for i, _ in enumerate(w_list)]  # 变量赋值
    return res  # 返回结果


def get_pdf_ratio(page_shape):  # 定义函数
    ratio = min(1600 / min(page_shape[0], page_shape[1]), 2500 / max(page_shape[0], page_shape[1]))  # 变量赋值
    return ratio  # 返回结果





def get_ocr(pdf_path):  # 定义函数
    ocr_data = []  # 变量赋值
    pdf = pdfplumber.open(pdf_path)  # 打开文件或资源
    shape_list = []  # 变量赋值
    for cc, page in enumerate(pdf.pages):  # 循环遍历
        if page.cropbox is None:  # 条件判断
            r_w, r_h = page.bbox[2], page.bbox[3]  # 变量赋值
            tpb_box = page.bbox  # 变量赋值
        else:  # 否则执行
            r_w, r_h = page.cropbox[2] - page.cropbox[0], page.cropbox[3] - page.cropbox[1]  # 变量赋值
            tpb_box = page.cropbox  # 变量赋值
        page_ratio = get_pdf_ratio((r_w, r_h))  # 变量赋值
        shape_list.append((int(r_w * page_ratio), int(r_h * page_ratio)))  # 添加元素到列表
        if 'char' in page.objects:  # 条件判断
            cluster_object = pdfplumber.utils.cluster_objects(page.objects['char'], "doctop", 4)  # 变量赋值
            ss = w_cluster_has_blank(cluster_object, (r_w, r_h), page_ratio, tpb_box)  # 变量赋值
        else:  # 否则执行
            ss = []  # 变量赋值
        # pdf_res_list.append(ss)
        tt = wrap_sorted_line_pdf(ss)  # 变量赋值
        ocr_data.append(tt)  # 添加元素到列表
    return ocr_data, shape_list  # 返回结果


def get_ocr_not_has_blank(pdf_path):  # 定义函数
    ocr_data = []  # 变量赋值
    pdf = pdfplumber.open(pdf_path)  # 打开文件或资源
    shape_list = []  # 变量赋值
    for cc, page in enumerate(pdf.pages):  # 循环遍历
        r_w, r_h = page.bbox[2], page.bbox[3]  # 变量赋值
        page_box = page.bbox  # 变量赋值
        shape_list.append((r_w, r_h))  # 添加元素到列表
        if 'char' in page.objects:  # 条件判断
            cluster_object = pdfplumber.utils.cluster_objects(page.objects['char'], "doctop", 4)  # 变量赋值
            ss = w_cluster(cluster_object, (r_w, r_h), 1.0, page_box)  # 变量赋值
        else:  # 否则执行
            ss = []  # 变量赋值
        tt = wrap_sorted_line_pdf(ss)  # 变量赋值
        ocr_data.append(tt)  # 添加元素到列表
    return ocr_data, shape_list  # 返回结果


def extract_text_allpage(filepath):  # 定义函数
    pdf = pdfplumber.open(filepath)  # 打开文件或资源
    for page in pdf.pages:  # 循环遍历
        tt = page.extract_text()  # 变量赋值

if __name__ == '__main__':  # 条件判断
    import time  # 导入时间处理模块
    sum_all = 0  # 变量赋值
    for i in range(100):  # 循环遍历
        t1 = time.time()  # 变量赋值
        aa, bb = get_ocr('/mnt/e/项目/小贝/回单易/EXIMBC101/需要标注/1_202503110915_1_1.pdf')  # 变量赋值
        t2 = time.time()  # 变量赋值
        sum_all += t2-t1  # 变量赋值
    # extract_text_allpage('/mnt/e/项目/cmbc_receipt.pdf')
    print(sum_all)
