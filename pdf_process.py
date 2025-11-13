import cv2
import numpy as np
import pdfplumber
import fitz
from collections import defaultdict
import copy
import time
import datetime


def two_to_four_convert(points, ratio):
    res = np.zeros([4, 2])
    res[0][0] = points[0] * ratio
    res[0][1] = points[1] * ratio
    res[1][0] = points[2] * ratio
    res[1][1] = points[1] * ratio
    res[2][0] = points[2] * ratio
    res[2][1] = points[3] * ratio
    res[3][0] = points[0] * ratio
    res[3][1] = points[3] * ratio
    return res


def  w_cluster_has_blank(cluster_obj, r_shape, ratio, crop_box=None):
    r_w, r_h = r_shape
    res = []
    for i in range(len(cluster_obj)):
        resi = []
        obj_line = cluster_obj[i]
        if len(obj_line) == 0:
            continue
        tmp_list = []
        tmp_text = ''
        span = []
        ts = sorted(obj_line, key=lambda a: a['x0'])
        # ts = obj_line
        ts = move_overlap(ts)
        for obj in ts:

            if not crop_box is None:
                if obj['x0'] <= crop_box[0] or obj['x0'] >= crop_box[2] or obj['y0'] <= crop_box[1] or obj['y1'] >= \
                        crop_box[3]:
                    continue

                obj['x0'] -= crop_box[0]
                obj['x1'] -= crop_box[0]
                obj['y0'] -= crop_box[1]
                obj['y1'] -= crop_box[1]

            if tmp_list == []:
                if obj['text'] == '':
                    continue
                tmp_list.append((obj['x0'], obj['y0'], obj['x1'], obj['y1']))
                tmp_text += obj['text']
                span.append(((obj['x0'] + obj['x1']) / 2 * ratio, (r_h - (obj['y0'] + obj['y1']) / 2) * ratio))
            else:
                # assert obj['x0']+0.5 >= tmp_list[-1][2]
                bb = obj['x0'] - tmp_list[-1][2]
                if abs(bb) <= 2:
                    # if obj['x0'] - tmp_list[-1][2] <= 6:
                    if obj['text'] == '':
                        continue
                    tmp_list.append((obj['x0'], obj['y0'], obj['x1'], obj['y1']))
                    tmp_text += obj['text']
                    span.append(((obj['x0'] + obj['x1']) / 2 * ratio, (r_h - (obj['y0'] + obj['y1']) / 2) * ratio))
                else:
                    tmp_blob = {
                        # 'box': [tmp_list[0][0], r_h - tmp_list[0][1], tmp_list[-1][2], r_h - tmp_list[-1][3]],
                        'box': two_to_four_convert(
                            [tmp_list[0][0], r_h - tmp_list[-1][3], tmp_list[-1][2], r_h - tmp_list[0][1]], ratio),
                        'text': tmp_text,
                        'span': span

                    }
                    resi.append(tmp_blob)
                    tmp_list = []
                    tmp_text = ''
                    span = []
                    if obj['text'] == '':
                        continue
                    tmp_list.append((obj['x0'], obj['y0'], obj['x1'], obj['y1']))
                    tmp_text += obj['text']
                    span.append(((obj['x0'] + obj['x1']) / 2 * ratio, (r_h - (obj['y0'] + obj['y1']) / 2) * ratio))
        if not tmp_list == []:
            tmp_blob = {
                # 'box': [tmp_list[0][0], r_h - tmp_list[0][1], tmp_list[-1][2], r_h - tmp_list[-1][3]],
                'box': two_to_four_convert(
                    [tmp_list[0][0], r_h - tmp_list[-1][3], tmp_list[-1][2], r_h - tmp_list[0][1]], ratio),
                'text': tmp_text,
                'span': span
            }
            if tmp_text == '':
                pass
            else:
                resi.append(tmp_blob)
        res.append(resi)
    return res


def move_overlap(ts_line):
    # 删除一行加粗的重叠字符
    last_span = (0, 1)
    res = []
    for ts in ts_line:
        this_span = (ts['x0'], ts['x1'])
        if (this_span[1] - this_span[0]) < 0.01:
            continue
        iou = (min(this_span[1], last_span[1]) - max(this_span[0], last_span[0])) / (this_span[1] - this_span[0])
        if iou > 0.7:
            continue
        else:
            last_span = this_span
            res.append(ts)
    return res


def w_cluster(cluster_obj, r_shape, ratio, page_box):
    r_w, r_h = r_shape
    res = []
    for i in range(len(cluster_obj)):
        resi = []
        obj_line = cluster_obj[i]
        if len(obj_line) == 0:
            continue
        tmp_list = []
        tmp_text = ''
        span = []
        ts = sorted(obj_line, key=lambda a: a['x0'])
        ts = move_overlap(ts)
        for obj in ts:
            if obj['x0'] < -10 + page_box[0] or obj['y0'] < -10+page_box[1] or obj['x1'] > page_box[2] + 10 or obj['y1'] > page_box[3] + 10:
                # print('aaa')
                continue
            if tmp_list == []:
                if obj['text'] == '' or obj['text'] == ' ':
                    continue
                tmp_list.append((obj['x0'], obj['y0'], obj['x1'], obj['y1']))
                tmp_text += obj['text']
                span.append(((obj['x0'] + obj['x1']) / 2 * ratio, (r_h - (obj['y0'] + obj['y1']) / 2) * ratio))
                continue
            else:
                # assert obj['x0']+0.5 >= tmp_list[-1][2]
                if obj['x0'] - tmp_list[-1][2] <= 8:
                    if obj['text'] == '' or obj['text'] == ' ':
                        continue
                    tmp_list.append((obj['x0'], obj['y0'], obj['x1'], obj['y1']))
                    tmp_text += obj['text']
                    span.append(((obj['x0'] + obj['x1']) / 2 * ratio, (r_h - (obj['y0'] + obj['y1']) / 2) * ratio))
                else:
                    tmp_blob = {
                        # 'box': [tmp_list[0][0], r_h - tmp_list[0][1], tmp_list[-1][2], r_h - tmp_list[-1][3]],
                        'box': two_to_four_convert(
                            [tmp_list[0][0], r_h - tmp_list[-1][3], tmp_list[-1][2], r_h - tmp_list[0][1]], ratio),
                        'text': tmp_text,
                        'span': span

                    }
                    resi.append(tmp_blob)
                    tmp_list = []
                    tmp_text = ''
                    span = []
                    if obj['text'] == '' or obj['text'] == ' ':
                        continue
                    tmp_list.append((obj['x0'], obj['y0'], obj['x1'], obj['y1']))
                    tmp_text += obj['text']
                    span.append(((obj['x0'] + obj['x1']) / 2 * ratio, (r_h - (obj['y0'] + obj['y1']) / 2) * ratio))
        if not tmp_list == []:
            tmp_blob = {
                # 'box': [tmp_list[0][0], r_h - tmp_list[0][1], tmp_list[-1][2], r_h - tmp_list[-1][3]],
                'box': two_to_four_convert(
                    [tmp_list[0][0], r_h - tmp_list[-1][3], tmp_list[-1][2], r_h - tmp_list[0][1]], ratio),
                'text': tmp_text,
                'span': span
            }
            resi.append(tmp_blob)
        res.append(resi)
    return res


def get_r_ratio(image_info):
    max_ratio = 1
    for image_info_i in image_info:
        tmp_h, tmp_w = image_info_i['bbox'][3] - image_info_i['bbox'][1], image_info_i['bbox'][2] - \
                       image_info_i['bbox'][0]
        g_h, g_w = image_info_i["height"], image_info_i["width"]
        ratio = max(g_w / tmp_w, g_h / tmp_h)
        if ratio > max_ratio and ratio <= 5:
            max_ratio = ratio
    return max_ratio


def get_rotate_angle(der_matrix):
    res = 0
    dist = 100
    rot_list = [0, 90, 180, 270]
    for rot in rot_list:
        rot_matrix = fitz.Matrix(rot)
        dist_i = abs(der_matrix[0] - rot_matrix[0]) + abs(der_matrix[1] - rot_matrix[1]) + abs(
            der_matrix[2] - rot_matrix[2]) + abs(der_matrix[3] - rot_matrix[3])
        if dist_i < dist:
            dist = dist_i
            res = rot
    return res


def ReadPdfImage(filepath=None, savepath=None, **kwargs):
    '''
    读取pdf中的图片，返回图片数组，格式为pil 图片类型
    :param filepath:
    :param pages:
    :param kwargs:
    :return:
    '''
    doc = fitz.open(filepath)
    images = []
    for i, page in enumerate(doc):
        canvos_point1 = (round(page.CropBox.top_left[0]), round(page.CropBox.top_left[1]))
        canvos_point2 = (round(page.CropBox.bottom_right[0]), round(page.CropBox.bottom_right[1]))
        canvos_w, canvos_h = canvos_point2[0] - canvos_point1[0], canvos_point2[1] - canvos_point1[1]
        canvos = np.ones([canvos_h, canvos_w, 3]) * 255
        image_info = page.get_image_info(xrefs=True)

        # shrink = fitz.Matrix(1 / 2340, 0, 0, 1 / 1654, 0, 0)
        # imgrect = fitz.Rect(0, 0, 2340, 1654)
        # bbox_dd, transform_dd = page.get_image_bbox('Im1', transform=True)
        # cc = shrink * transform_dd
        image_list = page.get_images()
        xref_list = {st['xref']: st for st in image_info}
        ratio = get_r_ratio(image_info)
        canvos = cv2.resize(canvos, None, None, ratio, ratio)
        s_list = []

        img_name_num_map = defaultdict(int)
        for im_info_i in image_list:
            name_i = im_info_i[7]
            img_name_num_map[name_i] += 1

        for c, item in enumerate(image_list):
            try:
                xref = item[0]
                if not xref in xref_list:
                    continue
                item_name = item[7]
                if img_name_num_map[item_name] >= 2:
                    continue
                # todo 这边可能重名， 从而影响到
                box, transform_ed = page.get_image_bbox(item_name, transform=True)
                point1 = (round(box.top_left.x), round(box.top_left.y))
                point2 = (round(box.bottom_right.x), round(box.bottom_right.y))
                pix = fitz.Pixmap(doc, xref)
                # ddf = pix.colorspace
                # ccs = pix.is_monochrome
                colorspace = xref_list[xref]['colorspace']

                tt = pix.tobytes()
                # nparr = np.fromstring(img['image'], np.uint8)
                nparr = np.fromstring(tt, np.uint8)
                img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if colorspace == 0:
                    img_decode = 255 - img_decode
                # cv2.imwrite('/mnt/e/data/ocr/保单/visual2/{}_{}_aa.png'.format(i, c), img_decode)
                transform = xref_list[xref]['transform']
                if transform[0] < 0:
                    img_decode = np.flip(img_decode, axis=1)
                if transform[3] < 0:
                    img_decode = np.flip(img_decode, axis=0)

                if point1[0] < -1 or point1[1] < -1 or point2[0] < -1 or point2[1] < -1:
                    new_point2 = (point2[0] - point1[0], point2[1] - point1[1])
                    new_point1 = (0, 0)
                    point1 = new_point1
                    point2 = new_point2

                ratio_s = img_decode.shape[1] / (point2[0] - point1[0])  # todo 这里用w的ratio更合理
                s_list.append((img_decode.shape[0] / (point2[1] - point1[1]),
                               img_decode.shape[1] / (point2[0] - point1[0]), point1, point2, img_decode.shape))
                d_ratio = ratio / ratio_s
                img_decode = cv2.resize(img_decode, None, None, d_ratio, d_ratio, cv2.INTER_CUBIC)
                point1 = (round(point1[0] * ratio), round(point1[1] * ratio))
                point2 = (round(point2[0] * ratio), round(point2[1] * ratio))
                # img_decode =  cv2.resize(img_decode, (point2[0]-point1[0], point2[1]-point1[1]))
                # max_h = min(canvos.shape[0]-point1[1], point2[1]-point1[1], img_decode.shape[0])
                # max_w = min(canvos.shape[1]- point1[0], point2[0]- point1[0], img_decode.shape[1])
                max_h = min(canvos.shape[0] - point1[1], img_decode.shape[0])
                max_w = min(canvos.shape[1] - point1[0], img_decode.shape[1])
                canvos[point1[1]:point1[1] + max_h, point1[0]:point1[0] + max_w, :] = img_decode[0:max_h, 0:max_w, :]
            except:
                continue
        rotate_angle = get_rotate_angle(page.derotation_matrix)
        r_canvos = np.rot90(canvos, rotate_angle // 90)
        images.append(r_canvos)
    return images


def wrap_sorted_line_pdf(sorted_line):
    res = {}
    res['span'] = []
    res['text'] = []
    res['ind'] = []
    res['blob'] = []
    for s, single_line in enumerate(sorted_line):
        line_text = ""
        position_list = []
        tmp_ind_list = []
        res_single_line = []
        ind_sum = 0
        for blob in single_line:
            # def blob_box_wrap(box):
            #     return np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
            # blob['box'] = blob_box_wrap(blob['box'])
            blob['text'] = blob['text'].replace('（', '(').replace('）', ')').replace('，', ',')
            if blob['text'] == '':
                continue
            kk = {
                'box': blob['box'],
                'text': blob['text']
            }
            line_text += blob['text']
            # tmp = Span2SinTextPoint_raw(blob['box'], blob['span'])
            position_list.extend(blob['span'])
            ind_sum += len(blob['text'])
            tmp_ind_list.append(ind_sum)
            res_single_line.append(kk)
        if line_text.replace(' ', '') == '':
            continue
        res['span'].append(position_list)
        res['text'].append(line_text)
        res['ind'].append(tmp_ind_list)
        res['blob'].append(res_single_line)
        res['raw_text'] = copy.deepcopy(res['text'])
    return res


def Span2SinTextPoint_raw(cnt, span):
    h = cnt[3][1] - cnt[0][1]
    h_start = (cnt[0][1] + cnt[3][1]) / 2
    w_start = cnt[0][0]
    w_l = cnt[1][0] - cnt[0][0]
    h_off_set = cnt[1][1] - cnt[0][1]
    # tan_theta = abs((cnt[1][1]-cnt[1][0])/(cnt[1][0] - cnt[0][0]))

    w_list = [int(w_start + (h / 32.0) * 8 * p - h / 32.0 * 30) for p in span]

    h_list = [int(h_start + h_off_set * (p / w_l)) for p in span]

    res = [(w_list[i], h_list[i]) for i, _ in enumerate(w_list)]
    return res


def get_pdf_ratio(page_shape):
    ratio = min(1600 / min(page_shape[0], page_shape[1]), 2500 / max(page_shape[0], page_shape[1]))
    return ratio





def get_ocr(pdf_path):
    ocr_data = []
    pdf = pdfplumber.open(pdf_path)
    shape_list = []
    for cc, page in enumerate(pdf.pages):
        if page.cropbox is None:
            r_w, r_h = page.bbox[2], page.bbox[3]
            tpb_box = page.bbox
        else:
            r_w, r_h = page.cropbox[2] - page.cropbox[0], page.cropbox[3] - page.cropbox[1]
            tpb_box = page.cropbox
        page_ratio = get_pdf_ratio((r_w, r_h))
        shape_list.append((int(r_w * page_ratio), int(r_h * page_ratio)))
        if 'char' in page.objects:
            cluster_object = pdfplumber.utils.cluster_objects(page.objects['char'], "doctop", 4)
            ss = w_cluster_has_blank(cluster_object, (r_w, r_h), page_ratio, tpb_box)
        else:
            ss = []
        # pdf_res_list.append(ss)
        tt = wrap_sorted_line_pdf(ss)
        ocr_data.append(tt)
    return ocr_data, shape_list


def get_ocr_not_has_blank(pdf_path):
    ocr_data = []
    pdf = pdfplumber.open(pdf_path)
    shape_list = []
    for cc, page in enumerate(pdf.pages):
        r_w, r_h = page.bbox[2], page.bbox[3]
        page_box = page.bbox
        shape_list.append((r_w, r_h))
        if 'char' in page.objects:
            cluster_object = pdfplumber.utils.cluster_objects(page.objects['char'], "doctop", 4)
            ss = w_cluster(cluster_object, (r_w, r_h), 1.0, page_box)
        else:
            ss = []
        tt = wrap_sorted_line_pdf(ss)
        ocr_data.append(tt)
    return ocr_data, shape_list


def extract_text_allpage(filepath):
    pdf = pdfplumber.open(filepath)
    for page in pdf.pages:
        tt = page.extract_text()

if __name__ == '__main__':
    import time
    sum_all = 0
    for i in range(100):
        t1 = time.time()
        aa, bb = get_ocr('/mnt/e/项目/小贝/回单易/EXIMBC101/需要标注/1_202503110915_1_1.pdf')
        t2 = time.time()
        sum_all += t2-t1
    # extract_text_allpage('/mnt/e/项目/cmbc_receipt.pdf')
    print(sum_all)
