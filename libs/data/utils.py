import math
import mmcv
import numpy as np
from numpy import random
import cv2
import numpy as np
import pandas as pd


def evaluate(values):
    average_ratio = []
    for (gt, pred) in values:
        ratio = abs(gt - pred) / gt
        average_ratio.append(ratio)
    return np.mean(average_ratio)

def get_interval_number(height_list, bins):
    seg = pd.cut(height_list, bins)
    counts = pd.value_counts(seg, sort=True)
    return counts[0]

def norm_by_height(img, img_save_path=None):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.array(gray_img, dtype=np.uint8)
    shape = gray_img.shape
    max_shape = max(shape[0], shape[1])
    if max_shape > 3000:
        size = 6
    elif max_shape > 2000:
        size = 4
    elif max_shape > 1000:
        size = 2
    else:
        size = 1
    height, width = shape[0] // size, shape[1] // size
    gray_img = cv2.resize(gray_img, (width, height))
    shape = gray_img.shape
    width_div_height = shape[1] / shape[0]
    if width_div_height > 1.5:
        width_ratio, height_ratio = 20, 10
    elif width_div_height < 0.67:
        width_ratio, height_ratio = 10, 20
    else:
        width_ratio, height_ratio = 10, 10
    ratio0, ratio1 = shape[0] // height_ratio, shape[1] // width_ratio

    is_break = False
    threshold_num = 120
    # import pdb; pdb.set_trace()
    while True:
        ret, bin_img = cv2.threshold(gray_img, threshold_num, 255, cv2.THRESH_BINARY)
        bin_img = 255 - bin_img
        contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours_array = np.array(contours)
        contours_filter = [ c for c in contours if c.shape[0]>3 ]
        if (len(contours_filter)) > 5:
            map_function = lambda x: cv2.boundingRect(x)
            rect_array = np.array(list(map(map_function, contours_filter)))
            contours_index = np.where((rect_array[:, 2] > 5) & (rect_array[:, 3] > 5) & (rect_array[:, 2] < ratio1) & (rect_array[:, 3] < ratio0))
            rects = rect_array[contours_index]

            height_list = rects[:, 3]
            if height_list.size > 5:
                value_count = pd.value_counts(height_list)
                if value_count.size < 3:
                    index_value = value_count.index
                else:
                    index_value = value_count.index[:3]
                bins_list = [(i-2, i+1) for i in index_value]
                interval_count_list = [get_interval_number(height_list, bins) for bins in bins_list]
                result = index_value[np.argmax(interval_count_list)] * size

                # show the image
                if img_save_path:
                    img = cv2.resize(img, ((width, height)))
                    for rect in rects:
                        x1, y1, w, h = rect
                        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (255, 0, 255), 1)
                    cv2.imwrite(img_save_path, img)

                return result

        if is_break:
            return 10 * size      # if it can't find the height, return the default value

        threshold_num += 20
        if threshold_num > 180:
            is_break = True

class InvalidFormat(Exception):
    pass


def segmentation_to_bbox(segmentation):
    x1 = min([pt[0] for contour in segmentation for pt in contour])
    y1 = min([pt[1] for contour in segmentation for pt in contour])
    x2 = max([pt[0] for contour in segmentation for pt in contour])
    y2 = max([pt[1] for contour in segmentation for pt in contour])
    return (x1, y1, x2, y2)


def cal_cell_bbox(table):
    cells_bbox = list()
    for cell in table['cells']:
        if 'segmentation' not in cell:
            cell_bbox = None
        else:
            segmentation = list()
            if 'sublines' in cell:
                for subline in cell['sublines']:
                    segmentation.extend(subline['segmentation'])
            if len(segmentation) == 0:
                segmentation = cell['segmentation']
            if len(segmentation) == 0:
                cell_bbox = None
            else:
                cell_bbox = segmentation_to_bbox(segmentation)
        cells_bbox.append(cell_bbox)
    return cells_bbox


def cal_cell_spans(table):
    layout = table['layout']
    num_cells = len(table['cells'])
    cells_span = list()
    for cell_id in range(num_cells):
        cell_positions = np.argwhere(layout == cell_id)
        y1 = np.min(cell_positions[:, 0])
        y2 = np.max(cell_positions[:, 0])
        x1 = np.min(cell_positions[:, 1])
        x2 = np.max(cell_positions[:, 1])
        assert np.all(layout[y1:y2, x1:x2] == cell_id)
        cells_span.append([x1, y1, x2, y2])
    return cells_span


def cal_fg_bg_span(spans, edge):
    num_span = len(spans)
    bg_spans = list()
    for idx in range(num_span):
        if spans[idx] is None:
            continue
        if idx == 0:
            if spans[idx][0] <= 0:
                continue
        else:
            if spans[idx-1] is None:
                continue
            if spans[idx][0] <= spans[idx-1][1]:
                continue
        if idx == num_span - 1:
            if spans[idx][1] >= edge:
                continue
        else:
            if spans[idx+1] is None:
                continue
            if spans[idx][1] >= spans[idx+1][0]:
                continue

        bg_spans.append(spans[idx])

    fg_spans = list()
    for idx in range(num_span+1):
        if idx == 0:
            s = 0
        else:
            if spans[idx-1] is None:
                continue
            s = spans[idx-1][1]

        if idx == num_span:
            e = edge
        else:
            if spans[idx] is None:
                continue
            e = spans[idx][0]

        if e <= s:
            continue

        fg_spans.append([s, e])

    return fg_spans, bg_spans


def shrink_spans(spans, size):
    new_spans = list()
    for idx, (start, end) in enumerate(spans):
        if idx == 0:
            if start <= 0:
                start = 1
        else:
            _, pre_end = spans[idx - 1]
            if start <= pre_end:
                shrink_distance = pre_end - start + 1
                start = start + math.ceil(shrink_distance / 2)

        if idx == len(spans) - 1:
            if end >= size:
                end = size - 1
        else:
            next_start, _ = spans[idx + 1]
            if end >= next_start:
                shrink_distance = end - next_start + 1
                end = end - math.ceil(shrink_distance / 2)
        if end - start < 1:
            raise InvalidFormat()

        new_spans.append([start, end])
    return new_spans


def cal_row_span(table, cells_span, cells_bbox, height):
    layout = table['layout']
    rows_span = list()
    for row_idx in range(layout.shape[0]):
        row = layout[row_idx, :]
        y1s = list()
        y2s = list()
        for cell_id in row:
            cell_span = cells_span[cell_id]
            cell_bbox = cells_bbox[cell_id]
            if (cell_span[1] == row_idx) and (cell_bbox is not None):
                y1s.append(cell_bbox[1])
            if (cell_span[3] == row_idx) and (cell_bbox is not None):
                y2s.append(cell_bbox[3])

        if (len(y1s) > 0) and (len(y2s) > 0):
            y1 = min(max(1, min(y1s)), height-1)
            y2 = min(max(1, max(y2s) + 1), height-1)
            rows_span.append([y1, y2])
        else:
            raise InvalidFormat()
    rows_span = shrink_spans(rows_span, height)
    rows_fg_span, rows_bg_span = cal_fg_bg_span(rows_span, height)
    return rows_fg_span, rows_bg_span


def cal_col_span(table, cells_span, cells_bbox, width):
    layout = table['layout']
    cols_span = list()
    for col_idx in range(layout.shape[1]):
        col = layout[:, col_idx]
        x1s = list()
        x2s = list()
        for cell_id in col:
            cell_span = cells_span[cell_id]
            cell_bbox = cells_bbox[cell_id]
            if (cell_span[0] == col_idx) and (cell_bbox is not None):
                x1s.append(cell_bbox[0])
            if (cell_span[2] == col_idx) and (cell_bbox is not None):
                x2s.append(cell_bbox[2])

        if (len(x1s) > 0) and (len(x2s) > 0):
            x1 = min(max(1, min(x1s)), width-1)
            x2 = min(max(1, max(x2s) + 1), width-1)
            cols_span.append([x1, x2])
        else:
            raise InvalidFormat()
    cols_span = shrink_spans(cols_span, width)
    cols_fg_span, cols_bg_span = cal_fg_bg_span(cols_span, width)
    return cols_fg_span, cols_bg_span


def extract_fg_bg_spans(table, image_size):
    width, height = image_size
    cells_bbox = cal_cell_bbox(table)
    cells_span = cal_cell_spans(table)
    # cal rows fg bg span
    rows_fg_span, rows_bg_span = cal_row_span(
        table, cells_span, cells_bbox, height
    )
    # cal cols fg bg span
    cols_fg_span, cols_bg_span = cal_col_span(
        table, cells_span, cells_bbox, width
    )
    return rows_fg_span, rows_bg_span, cols_fg_span, cols_bg_span

def random_sample_ratio(img_scale, ratio):
    """Randomly sample an img_scale when ``ratio_range`` is specified.

    A ratio will be randomly sampled from the range specified by
    ``ratio_range``. Then it would be multiplied with ``img_scale`` to
    generate sampled scale.

    Args:
        img_scale (tuple): Images scale base to multiply with ratio.
        ratio_range (tuple[float]): The minimum and maximum ratio to scale
            the ``img_scale``.

    Returns:
        scale: scale is sampled ratio multiplied with ``img_scale``
    """
    scale = (int(img_scale[0] * ratio), int(img_scale[1] * ratio))
    return scale

class Resize:
    def __init__(self, ratio_range, keep_ratio=True, bool_training=False, max_size=1024, img_size_dict=None, height_norm=16, min_size=64, bbox_clip_border=True, backend='cv2', override=False):
        self.backend = backend
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.max_size = max_size
        self.min_size = min_size
        self.override = override
        self.bbox_clip_border = bbox_clip_border
        self.bool_training = bool_training
        self.img_size_dict = img_size_dict
        self.height_norm = height_norm


    def _random_scale(self, results):
        img_sacle = results['img'].shape[:2][::-1]#w,h


        if self.bool_training:
            scale = self.img_size_dict[results['image_path']]
            scale = max(scale[0], self.min_size), max(scale[1], self.min_size)
        else:
            if results['image_path'] in self.img_size_dict.keys():
                scale = self.img_size_dict[results['image_path']]
                scale = max(scale[0], self.min_size), max(scale[1], self.min_size)
                # scale = (1000,1000)
            else:
                height = norm_by_height(results['img'])
                height_ratio = self.height_norm / height
                scale = (int(img_sacle[0]*height_ratio), int(img_sacle[1]*height_ratio))
                scale = max(scale[0], self.min_size), max(scale[1], self.min_size)

        # if max(scale) > self.max_size:
        #     ratio = min(self.max_size/scale[0], self.max_size/scale[1])
        #     scale = int(scale[0] * ratio), int(scale[1] * ratio)

        # if self.bool_training:
        #     scale = max(min(self.max_size, scale[0]), self.min_size), max(min(self.max_size, scale[1]), self.min_size)
        results['scale'] = scale
        results['scale_rate'] = (img_sacle[0]/scale[0], img_sacle[0]/scale[0])

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in ['img']:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w # 新的图片相较于原先的比例
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            results[key] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_labels(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        img_shape = results['img_shape']
        for key in ['row_start_center_bboxes', 'col_start_center_bboxes']:
            bboxes = np.array(results[key], dtype=np.float32) * results['scale_factor']
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes
        for key in ['row_line_segmentations', 'col_line_segmentations']:
            polygons = list()
            for poly in results[key]:
                poly = np.array(poly, dtype=np.float32).reshape(-1, 2) # (N, 2)
                poly[:, 0] *= results['scale_factor'][0] # (N, 2)
                poly[:, 1] *= results['scale_factor'][1] # (N, 2)
                if self.bbox_clip_border:
                    poly[:, 0] = np.clip(poly[:, 0], 0, img_shape[1]) # (N, 2)
                    poly[:, 1] = np.clip(poly[:, 1], 0, img_shape[0]) # (N, 2)
                polygons.append(poly)
            results[key] = polygons

        for key in ['row_line_points', 'col_line_points']:
            polygons = list()
            for poly in results[key]:
                poly = np.array(poly, dtype=np.float32).reshape(-1, 2) # (N, 2)
                poly[:, 0] *= results['scale_factor'][0] # (N, 2)
                poly[:, 1] *= results['scale_factor'][1] # (N, 2)
                if self.bbox_clip_border:
                    poly[:, 0] = np.clip(poly[:, 0], 0, img_shape[1]) # (N, 2)
                    poly[:, 1] = np.clip(poly[:, 1], 0, img_shape[0]) # (N, 2)
                polygons.append(poly)
            results[key] = polygons

    def __call__(self, results):
        results["rota_angle"] = self.cal_rota_angle(results["rota_angle"])
        # results["slope"] = self.cal_slope( results["rota_angle"] )

        self._random_scale(results)
        self._resize_img(results)
        self._resize_labels(results)
        results['image_h'], results['image_w'], _ = results['img'].shape
        return results

    def cal_rota_angle(self,angle):
        if np.isnan(angle):
            angle = 0
        # angle = max( min(angle,20),-20 )
        # angle = max( min(angle,20),-20 )
        return angle

    def cal_slope(self, angle):
        angle_rad = angle/180*math.pi
        slope = np.tan(angle_rad)
        return slope


class Resize_iflytab:
    def __init__(self, ratio_range, keep_ratio=True, bool_training=False, max_size=1024, img_size_dict=None, height_norm=16, min_size=64, bbox_clip_border=True, backend='cv2', override=False):
        self.backend = backend
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.max_size = max_size
        self.min_size = min_size
        self.override = override
        self.bbox_clip_border = bbox_clip_border
        self.bool_training = bool_training
        self.img_size_dict = img_size_dict
        self.height_norm = height_norm

    def _random_scale(self, results):
        img_sacle = results['img'].shape[:2][::-1]#w,h


        if self.bool_training:
            scale = self.img_size_dict[results['image_path']]
            scale = max(scale[0], self.min_size), max(scale[1], self.min_size)
        else:
            if results['image_path'] in self.img_size_dict.keys():
                scale = self.img_size_dict[results['image_path']]
                scale = max(scale[0], self.min_size), max(scale[1], self.min_size)
                # scale = (1000,1000)
            else:
                height = norm_by_height(results['img'])
                height_ratio = self.height_norm / height
                scale = (int(img_sacle[0]*height_ratio), int(img_sacle[1]*height_ratio))
                scale = max(scale[0], self.min_size), max(scale[1], self.min_size)

        # if max(scale) > self.max_size:
        #     ratio = min(self.max_size/scale[0], self.max_size/scale[1])
        #     scale = int(scale[0] * ratio), int(scale[1] * ratio)

        # if self.bool_training:
        #     scale = max(min(self.max_size, scale[0]), self.min_size), max(min(self.max_size, scale[1]), self.min_size)
        results['scale'] = scale
        results['scale_rate'] = (img_sacle[0]/scale[0], img_sacle[0]/scale[0])

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in ['img']:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w # 新的图片相较于原先的比例
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            results[key] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_labels(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        img_shape = results['img_shape']
        for key in ['row_start_center_bboxes', 'col_start_center_bboxes']:
            bboxes = np.array(results[key], dtype=np.float32) * results['scale_factor']
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes
        for key in ['row_line_segmentations', 'col_line_segmentations']:
            polygons = list()
            for poly in results[key]:
                poly = np.array(poly, dtype=np.float32).reshape(-1, 2) # (N, 2)
                poly[:, 0] *= results['scale_factor'][0] # (N, 2)
                poly[:, 1] *= results['scale_factor'][1] # (N, 2)
                if self.bbox_clip_border:
                    poly[:, 0] = np.clip(poly[:, 0], 0, img_shape[1]) # (N, 2)
                    poly[:, 1] = np.clip(poly[:, 1], 0, img_shape[0]) # (N, 2)
                polygons.append(poly)
            results[key] = polygons
        for key in ['row_split_lines','col_split_lines']:
            polygons = list()
            for poly in results[key]:
                poly = np.array(poly, dtype=np.float32).reshape(-1, 2) # (N, 2)
                poly[:, 0] *= results['scale_factor'][0] # (N, 2)
                poly[:, 1] *= results['scale_factor'][1] # (N, 2)
                if self.bbox_clip_border:
                    poly[:, 0] = np.clip(poly[:, 0], 0, img_shape[1]) # (N, 2)
                    poly[:, 1] = np.clip(poly[:, 1], 0, img_shape[0]) # (N, 2)
                polygons.append(poly)
            results[key] = polygons

    def __call__(self, results):

        self._random_scale(results)
        self._resize_img(results)
        self._resize_labels(results)
        results['image_h'], results['image_w'], _ = results['img'].shape
        return results

    def cal_rota_angle(self,angle):
        if np.isnan(angle):
            angle = 0
        # angle = max( min(angle,20),-20 )
        # angle = max( min(angle,20),-20 )
        return angle

    def cal_slope(self, angle):
        angle_rad = angle/180*math.pi
        slope = np.tan(angle_rad)
        return slope


class PhotoMetricDistortion:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        img = img.astype(np.float32)
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img