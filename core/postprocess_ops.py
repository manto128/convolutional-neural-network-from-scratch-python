import numpy as np
from core.kernel_ops import Operator
from core.utils import roundupdown

class Quantize(Operator):
  def __init__(self):
    super().__init__()

  def forward(self):
    if self.input_port[0].data.dtype is np.float32:
      self.output_port.send(np.int8(roundupdown(self.input_port[0].data / self.o_scales + self.o_zero_points)))
    elif self.input_port[0].data.dtype is np.int8:
      self.output_port.send(np.uint8(roundupdown((self.input_port[0].data - self.i_zero_points)/ self.o_scales) + self.o_zero_points))

  def feed(self, o_scales=None, o_zero_points=None,
            i_scales=None, i_zero_points=None):
    self.o_scales = o_scales
    self.o_zero_points = o_zero_points
    self.i_scales = i_scales
    self.i_zero_points = i_zero_points

class Dequantize(Operator):
  def __init__(self):
    super().__init__()

  def forward(self):
    self.output_port.send((self.input_port[0].data - self.zero_point) * self.scale)

  def feed(self, scale, zero_point):
    self.scale = scale
    self.zero_point = zero_point

class Detection_PostProcess(Operator):
  def __init__(self, 
               h_scale,
               w_scale, 
               x_scale, 
               y_scale, 
               use_regular_nms, 
               num_classes, 
               max_classes_per_detection, 
               max_detections, 
               nms_iou_threshold, 
               nms_score_threshold):
    super().__init__()
    self.h_scale = h_scale
    self.w_scale = w_scale
    self.x_scale = x_scale
    self.y_scale = y_scale
    self.use_regular_nms = use_regular_nms
    self.num_classes = num_classes
    self.max_classes_per_detection = max_classes_per_detection
    self.max_detections = max_detections
    self.nms_iou_threshold = nms_iou_threshold
    self.nms_score_threshold = nms_score_threshold

  def feed(self, anchors):
    """
    Args:
    - anchors: Anchor boxes used by the model (shape: [num_anchors, 4]).
    """
    self.anchors = anchors

  def forward(self):
    output_boxes = self.input_port[0].data
    output_scores = self.input_port[1].data

    # Decode the bounding boxes
    decoded_boxes = self.decode_boxes(output_boxes, self.anchors)

    # Apply non-max suppression
    selected_boxes, selected_scores = self.non_max_suppression(decoded_boxes[0], output_scores[0])

    # print("Selected boxes after NMS:")
    # print(selected_boxes)
    # print("Confidence scores for the selected boxes:")
    # print(selected_scores)

  def decode_boxes(self, output_boxes):
    """
    Decode the predicted bounding boxes.
    
    Args:
    - output_boxes: Predicted bounding boxes output by the model (shape: [batch_size, num_anchors, 4]).
    
    Returns:
    - decoded_boxes: Decoded bounding boxes (shape: [batch_size, num_anchors, 4]).
    """
    batch_size = output_boxes.shape[0]
    num_anchors = output_boxes.shape[1]

    decoded_boxes = np.zeros_like(output_boxes)

    for batch_idx in range(batch_size):
      for anchor_idx in range(num_anchors):
        y, x, h, w = output_boxes[batch_idx, anchor_idx]
        anchor_y, anchor_x, anchor_h, anchor_w = self.anchors[anchor_idx]

        y_center = y / self.y_scale * anchor_h + anchor_y
        x_center = x / self.x_scale * anchor_w + anchor_x
        half_h = 0.5*np.exp(h / self.h_scale) * anchor_h
        half_w = 0.5*np.exp(w / self.w_scale) * anchor_w
        y_min = y_center - half_h
        y_max = y_center + half_h
        x_min = x_center - half_w
        x_max = x_center + half_h

        decoded_boxes[batch_idx, anchor_idx] = [y_min, x_min, y_max, x_max]

    return decoded_boxes

  def non_max_suppression(self, boxes, scores):
    """
    Apply Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes.
    
    Args:
    - boxes: Decoded bounding boxes (shape: [num_boxes, 4]).
    - scores: Confidence scores for each box and class (shape: [num_boxes, num_classes]).
    
    Returns:
    - selected_boxes: Selected bounding boxes after NMS (shape: [num_selected_boxes, 4]).
    - selected_classes: Class labels for the selected boxes (shape: [num_selected_boxes]).
    - selected_scores: Confidence scores for the selected boxes (shape: [num_selected_boxes]).
    """
    if len(boxes) == 0:
        return [], [], []

    num_classes = scores.shape[2]
    selected_boxes = []
    selected_classes = []
    selected_scores = []

    for class_idx in range(num_classes):
      class_scores = scores[:, :, class_idx]
      class_mask = class_scores > self.nms_score_threshold

      class_boxes = boxes[class_mask]
      class_scores = class_scores[class_mask]

      if len(class_boxes) == 0:
          continue

      class_selected_boxes, class_selected_scores = self.select_boxes(class_boxes, class_scores)

      selected_boxes.extend(class_selected_boxes)
      selected_classes.extend([class_idx] * len(class_selected_boxes))
      selected_scores.extend(class_selected_scores)

    selected_boxes = np.array(selected_boxes)
    selected_classes = np.array(selected_classes)
    selected_scores = np.array(selected_scores)

    return selected_boxes, selected_classes, selected_scores

  def select_boxes(self, boxes, scores):
    selected_indices = np.argsort(scores)[::-1]
    selected_boxes = []
    selected_scores = []

    while len(selected_indices) > 0:
      current_index = selected_indices[0]
      selected_boxes.append(boxes[current_index])
      selected_scores.append(scores[current_index])

      iou = self.calculate_iou(boxes[current_index], boxes[selected_indices[1:]])
      overlap_indices = np.where(iou <= self.nms_iou_threshold)[0]
      selected_indices = selected_indices[overlap_indices + 1]

    return selected_boxes, selected_scores

  def calculate_iou(box, other_boxes):
    """
    Calculate Intersection over Union (IoU) between a box and a set of other boxes.
    
    Args:
    - box: A single box (format: [y_min, x_min, y_max, x_max]).
    - other_boxes: Other boxes to compare against (shape: [num_boxes, 4]).
    
    Returns:
    - iou: IoU values between the box and other boxes (shape: [num_boxes]).
    """
    y_min = np.maximum(box[0], other_boxes[:, 0])
    x_min = np.maximum(box[1], other_boxes[:, 1])
    y_max = np.minimum(box[2], other_boxes[:, 2])
    x_max = np.minimum(box[3], other_boxes[:, 3])

    intersection_area = np.maximum(0, y_max - y_min) * np.maximum(0, x_max - x_min)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    other_boxes_area = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])

    iou = intersection_area / (box_area + other_boxes_area - intersection_area)
    
    return iou