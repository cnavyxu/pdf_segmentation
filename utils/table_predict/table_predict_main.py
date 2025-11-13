# 导入 copy 模块，用于对象拷贝
import copy
# 导入 time 模块，用于时间相关操作
import time

# 导入表格标签和距离配置
from utils.table_predict.utils.local_settings import BORDERLESS_LABEL, LINED_LABEL, DIST_DICT, DIST_WHITE
# 导入表格位置检测模块
from utils.table_predict.predict.predict_location import TABLE_LOCATION
# 导入无线表格结构识别模块
from utils.table_predict.predict.predict_structure_borderless import TABLE_STRUCT_BORDERLESS
# 导入有线表格结构识别模块
from utils.table_predict.predict.predict_structure_lined import TABLE_STRUCT_LINED
# 导入有线表格后处理模块
from utils.table_predict.structure_lined_postprocessing import lined_postprocessing, p_bbox2structure
# 导入无线表格后处理模块
from utils.table_predict.structure_borderless_postprocessing import borderless_postprocessing, \
    p_slanet_aligning
# 导入工具函数模块
from utils.table_predict.utils import slanet_html_function, generate_function, p_pts_rect2poly
# 导入工具类
from utils.table_predict.utils.ttb_utils_ym import *


def fix_min_len_resize(img, min_l, max_len=2500):
    """
    按最小边长调整图像大小
    
    参数:
        img: 输入图像
        min_l: 最小边长
        max_len: 最大边长，默认 2500
    
    返回:
        res_img: 调整大小后的图像
        res_ratio: 缩放比例 (width_ratio, height_ratio)
    """
    # 获取图像的高度和宽度
    h, w = img.shape[0:2]
    # 计算基于最小边长的缩放比例
    ratio = float(min_l) / min(h, w)
    # 限制最大边长不超过 max_len
    ratio = min(max_len / max(h, w), ratio)
    # 计算新的高度和宽度
    new_h, new_w = int(ratio * h), int(ratio * w)
    # 调整图像大小
    res_img = cv2.resize(img, (new_w, new_h))
    # 计算缩放比例
    res_ratio = (float(w) / new_w, float(h) / new_h)
    # 返回调整后的图像和缩放比例
    return res_img, res_ratio


class TABLE():
    """表格识别主类"""
    
    def __init__(self, gpu_id, location_model_path, struct_lined_model_path):
        """
        初始化表格识别器
        
        参数:
            gpu_id: GPU 设备 ID
            location_model_path: 表格位置检测模型路径
            struct_lined_model_path: 有线表格结构识别模型路径
        """
        # 初始化表格位置检测器
        self.table_location = TABLE_LOCATION(
            gpu_id=gpu_id, model_path=location_model_path)
        # 备用：识别无线表格
        # self.table_struct_borderless = TABLE_STRUCT_BORDERLESS(
        #     gpu_id=gpu_id,
        #     model_path=struct_borderless_model_path,
        #     table_max_len=table_max_len,
        #     table_char_dict_path=table_char_dict_path,
        # )
        # 初始化有线表格结构识别器
        self.table_struct_lined = TABLE_STRUCT_LINED(
            gpu_id=gpu_id, model_path=struct_lined_model_path)
        # 测试模式标志
        self.test = False
                
    def merge(self, table_points, cut_points_list, dist_):
        """
        将截取表格中的单元格坐标转换为原图坐标
        
        参数:
            table_points: 表格在原图中的顶点坐标
            cut_points_list: 单元格在截取表格图中的坐标列表
            dist_: 截取表格时额外截取的边框宽度（有线表格为 64，无线表格为 0）
        
        返回:
            cell_bbox_list: 单元格在原图中的坐标列表
        """
        # 获取表格在原图中的最小 X 坐标
        min_w = min([pt[0] for pt in table_points])
        # 获取表格在原图中的最小 Y 坐标
        min_h = min([pt[1] for pt in table_points])
        # 设置宽度边距
        dist_iw = dist_
        # 设置高度边距
        dist_ih = dist_

        # 计算截取区域的起始 X 坐标（不小于 0）
        cut_iw = max(min_w - dist_iw, 0)
        # 计算截取区域的起始 Y 坐标（不小于 0）
        cut_ih = max(min_h - dist_ih, 0)

        # 初始化单元格坐标列表
        cell_bbox_list = []
        # 遍历所有单元格坐标
        for cut_index, cut_bbox in enumerate(cut_points_list):
            # 将截取图中的坐标转换为原图坐标
            cell_bbox = [[pt[0] + cut_iw, pt[1] + cut_ih] for pt in cut_bbox]
            # 添加到结果列表
            cell_bbox_list.append(cell_bbox)
        # 返回转换后的坐标列表
        return cell_bbox_list

    def table_predict(self, img, res_ocr_i=None):
        time_dict = {
            'location':[],
            'BORDERLESS':[],
            'LINED':[],
            'postprocessing':[],
            'timer':[],
            'timer_cpp':[],
        }
        if res_ocr_i is None:
            res_ocr_i = {
                'blob':[],
                'ind':[],
                'raw_text':[],
                'span':[],
                'text':[],
            }
        # 计算文字平均高度
        # word_h_mean = generate_function.f_get_h_mean(res_ocr_i)
        # word_h_mean = 25.161610233629492

        img_pdf, _ = fix_min_len_resize(img, 1012)
        img, _ = fix_min_len_resize(img, 1600)
        ratio = img_pdf.shape[0]/img.shape[0]
        # res_ocr_i_origin = kk_demo.f_get_res_ocr_i(img)
        res_ocr_i_origin = res_ocr_i

        t1 = time.time()
        location_result = self.table_location.predict(img)
        t2 = time.time()
        time_dict['location'].append(round(t2-t1, 4))

        shapes = []
        for table_i, table_result in enumerate(location_result):
            table_label = table_result['label']
            table_bbox = table_result['bbox']
            table_points = [[table_bbox[0], table_bbox[1]], [table_bbox[2], table_bbox[3]]]

            shape = {'table_points': table_points, 'table_label': table_label}
            shapes.append(shape)

        img_cut_list = generate_function.get_img_cut(img, shapes)
        
        contour_analyzer_list = []
        # for table_i, table_result in enumerate(location_result):
        for img_cut_info in img_cut_list:
            # table_label = table_result['label']
            # table_bbox = table_result['bbox']
            # table_points = [[table_bbox[0], table_bbox[1]], [table_bbox[2], table_bbox[3]]]

            table_i = img_cut_info['table_i']
            table_label = img_cut_info['table_label']
            table_points = img_cut_info['table_points']
            img_cut = img_cut_info['img_cut']
            
            blob = copy.deepcopy(res_ocr_i_origin['blob'])
            res_ocr_i = {
                'blob': blob
            }
            if table_label == BORDERLESS_LABEL:
                # 无线表格
                t1 = time.time()
                dist_ = DIST_DICT[table_label]
                # img_cut = self.get_img_cut(img, table_points, dist_=dist_)
                structure_str_list, cut_bbox_list = self.table_struct_borderless.predict(img_cut)
                cut_points_list = [p_pts_rect2poly.f_rect2poly(bbox) for bbox in cut_bbox_list]
                
                html_content = ''.join(structure_str_list)
                
                cell_points_list = self.merge(
                    table_points=table_points, cut_points_list=cut_points_list, dist_=dist_)

                # 对齐无线表格单元格坐标
                cell_list, has_merged_cells, column_num, row_num = p_slanet_aligning.f_aligning(
                    html_content=html_content, points_list=cell_points_list)

                # count_columns = COUNT_COLUMNS()
                # column_num, row_num = count_columns(structure_str_list)
                # row_col_center_id = borderless_postprocessing.f_get_row_col_center_id(
                #     column_num=column_num,
                #     row_num=row_num,
                # )
                
                contour_analyzer = borderless_postprocessing.f_borderless_postprocessing(
                    res_ocr_i=res_ocr_i,
                    cell_list=cell_list,
                )
                # row_col_center_id=row_col_center_id,
                # contour_analyzer.label = 1

                if has_merged_cells:
                    performance = 'bad'
                else:
                    performance = 'mediocre'
                
                t2 = time.time()
                time_dict['BORDERLESS'].append(round(t2 - t1, 4))
            elif table_label == LINED_LABEL:
                # 有线表格
                t1 = time.time()
                dist_ = DIST_DICT[table_label]
                # img_cut = self.get_img_cut(img, table_points, dist_=dist_)
                cut_bbox_list = self.table_struct_lined.predict(img_cut, ratio=ratio)
                # , timer, timer_cpp
                cut_points_list = [p_pts_rect2poly.f_rect2poly(bbox) for bbox in cut_bbox_list]
                
                cell_points_list = self.merge(
                    table_points=table_points, cut_points_list=cut_points_list, dist_=dist_)

                # cell_points_list = [p_pts_rect2poly.f_rect2poly(points) for points in cell_points_list]

                t_postprocessing_st = time.time()
                # 对齐有线表格
                # cell_list是包含表格结构的完整数据
                cell_list, row_num, column_num, box_dict_origin, box_dict_new, performance = \
                    p_bbox2structure.f_bbox2cell(cell_points_list)
                
                contour_analyzer = lined_postprocessing.f_lined_postprocessing(
                    res_ocr_i=res_ocr_i,
                    cell_list=cell_list,
                )
                # contour_analyzer.label = 2
                # if self.test:
                contour_analyzer.box_dict_origin = box_dict_origin
                contour_analyzer.box_dict_new = box_dict_new
                
                t2 = time.time()
                time_dict['postprocessing'].append(round(t2 - t_postprocessing_st, 4))
                time_dict['LINED'].append(round(t2 - t1, 4))
                # time_dict['timer'].append(timer)
                # time_dict['timer_cpp'].append(timer_cpp)
            else:
                continue

            # 无线表格cut_points_list未对齐，cut_points_list_aligning、cell_center_list、cell_hull_coord已对齐
            # 有线表格cut_points_list、cell_center_list未对齐，cell_hull_coord已对齐
            contour_analyzer.row_num = row_num
            contour_analyzer.column_num = column_num
            contour_analyzer.cell_list = cell_list

            html_list = slanet_html_function.f_cell2html(cell_list, need_text=True)
            contour_analyzer.html_list = html_list
            contour_analyzer.html = '<table border="1">'+''.join(html_list)+'</table>'
            
            blob = copy.deepcopy(res_ocr_i['blob'])
            contour_analyzer.res_ocr_i = {'blob': blob}
            contour_analyzer.img_origin = img
            contour_analyzer.table_label = table_label
            contour_analyzer.table_points = table_points
            contour_analyzer.performance = performance
            contour_analyzer.img_cut = img_cut
            contour_analyzer.cut_points_list = cut_points_list
            contour_analyzer_list.append(contour_analyzer)

        if self.test:
            generate_function.f_time_count(time_dict)
        return contour_analyzer_list#, time_dict
            
            
if __name__ == '__main__':
    import cv2

    # location_model_path = '/home/jiangweidong/table_demo/best.onnx'
    # struct_borderless_model_path = '/home/jiangweidong/table_demo/SLANet_ch_20241202/model.onnx'
    # table_char_dict_path = '/home/jiangweidong/table_demo/SLANet_ch_20241202/table_structure_dict_ch.txt'
    # struct_lined_model_path = '/home/jiangweidong/table_demo/table_line.onnx'
    # img_path_read = '/home/jiangweidong/table_demo/imgs/完税证明/cp.png'

    location_model_path = '/home/zhangym/deploy/yolov7-main/runs/train/yolov741/weights/best.onnx'
    # struct_borderless_model_path = '/home/zhangym/data/data_temp/slanet_data/model/SLANet_ch_20241202_2/model.onnx'
    struct_borderless_model_path = '/home/zhangym/data/data_temp/slanet_data/model/SLANet_ch_20241205_6_2/slim.onnx'
    table_char_dict_path = '/home/zhangym/deploy/ocr_table_20241125/paddleocr_local/ppocr/utils/dict/table_structure_dict_ch.txt'
    struct_lined_model_path = '/home/zhangym/data/data_end/pb/vin_detect_2024-11-05 15-03-40_034w.onnx'
    img_path_read = '/home/zhangym/data/data_origin/data_1113/完税证明20241128/img/1.png'
    
    table = TABLE(
        location_model_path=location_model_path, 
        struct_borderless_model_path=struct_borderless_model_path,
        struct_lined_model_path=struct_lined_model_path,
        table_char_dict_path=table_char_dict_path,
    )

    img = cv2.imread(img_path_read)
    contour_analyzer_list = table.table_predict(img)