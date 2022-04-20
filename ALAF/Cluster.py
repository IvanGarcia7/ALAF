import numpy as np

def rectangle_area(sq):
    """
    Function to calculate the area of a rectangle. Square is assomed to be given as a list: [min_height, min_width, max_height, max_width]
    Args:
        sq ([type]): [description]
    Returns:
        [type]: [description]
    """

    sq_min_h = sq[0]
    sq_max_h = sq[2]
    sq_min_w = sq[1]
    sq_max_w = sq[3]
    
    height = sq_max_h - sq_min_h
    
    if height < 0:
        return -1
    
    width = sq_max_w - sq_min_w
    
    if width < 0:
        return -1
    
    return height*width


def bbox_structure_to_square(bbox):
    ymin,xmin,ymax,xmax = bbox
    bbox_adecuada = [ymin,xmin,ymax,xmax]
    return bbox_adecuada

def get_most_confiable_object_idx(lists):
    max_confidence = None
    max_confidence_object_idx = None
    max_confidence_object_list_idx = None
    for l_idx, l in enumerate(lists):
        _, _, confidences = l
        current_list_max_confidence = confidences.max()
        if not max_confidence or max_confidence < current_list_max_confidence:
            max_confidence = current_list_max_confidence
            max_confidence_object_idx = np.where(confidences==max_confidence)[0][0]
            max_confidence_object_list_idx = l_idx

    return max_confidence_object_idx, max_confidence_object_list_idx


def union_of_two_rectangles(sq1, sq2):
    """
    Function to calculate the union area of two rectangle. Squares are assumed to be given as list: [min_height, min_width, max_height, max_width]   
    Args:
        sq1 ([type]): [description]
        sq2 ([type]): [description]
    Returns:
        [type]: [description]
    """

    sq1_area = rectangle_area(sq1)
    sq2_area = rectangle_area(sq2)
    
    return sq1_area + sq2_area - intersection_between_two_rectangles(sq1, sq2)    




def intersection_between_two_rectangles(sq1, sq2):
    """
    Function to calculate the intersection area between two rectangle. Squares are assumed to be given as list: [min_height, min_width, max_height, max_width]
    Args:
        sq1 ([type]): [description]
        sq2 ([type]): [description]
    Returns:
        [type]: [description]
    """

    sq1_min_h = sq1[0]
    sq1_max_h = sq1[2]
    sq1_min_w = sq1[1]
    sq1_max_w = sq1[3]
    
    sq2_min_h = sq2[0]
    sq2_max_h = sq2[2]
    sq2_min_w = sq2[1]
    sq2_max_w = sq2[3]
    
    # We will create the intersection rectangle.
    in_sq_min_h = max(sq1_min_h, sq2_min_h)
    in_sq_max_h = min(sq1_max_h, sq2_max_h)
    in_sq_min_w = max(sq1_min_w, sq2_min_w)
    in_sq_max_w = min(sq1_max_w, sq2_max_w)
    
    area = rectangle_area([in_sq_min_h, in_sq_min_w, in_sq_max_h, in_sq_max_w])
    
    if area != -1:
        return area
    else:
        return 0


def intersection_over_union_using_squares(sq1, sq2):
    """
    Function to calculate the intersection over union. Squares are assomed to be given as a list: [min_height, min_width, max_height, max_width] 
    Args:
        sq1 ([type]): [description]
        sq2 ([type]): [description]
    Returns:
        [type]: [description]
    """

    intersection = intersection_between_two_rectangles(sq1,sq2)
    union = union_of_two_rectangles(sq1,sq2)
    
    return intersection/union

def intersection_over_union(bbox1, bbox2):
    sq1 = bbox_structure_to_square(bbox1)
    sq2 = bbox_structure_to_square(bbox2)
    return intersection_over_union_using_squares(sq1, sq2)



def get_same_object(obj, detections_list, thd, ignore_class = True):
    """Function to get the nearest object to obj from list of object with minimum an IOU of thd.
    Args:
        obj (tuple(bbox, class, confidence)): obj to compare.
        detections_list (tuple(list(bbox), list(class), list(confidence))): List containing dected objects to compare obj.
        thd (float): Minimum iou to be a candidate.
        ignore_class (bool, optional): Do we ignore class to assign equivalence?. Defaults to True.
    Returns:
        int: idx from detections_list[0] with the greater iou compared to obj.
    """

    list_bboxes, list_classes, _ = detections_list
    obj_bbox, obj_class, _ = obj
    greater_iou = None
    greater_iou_idx = None

    for idx, (bbox, _class) in enumerate(zip(list_bboxes, list_classes)):
        iou = intersection_over_union(obj_bbox, bbox)
        if ignore_class or obj_class == _class:

            if iou > thd and ((not greater_iou) or iou > greater_iou):
                greater_iou = iou
                greater_iou_idx = idx

    return greater_iou_idx


def create_clusters(_lists, threshold):

    """Function to create clusters by assigning equivalences between N>1 lists of objects based on their bboxes IoU.
    Args:  
        _lists (list(tuple)) : An arbitrary list of tuples with structure (np.array(bbox), np.array(classs), np.array(confidences))
    Returns:
    """

    lists = _lists.copy()


    empty_lists_to_pop_idx = []
    for idx, _list in enumerate(lists):
        _, classes, _ = _list
        if classes.shape[0] ==0:
            empty_lists_to_pop_idx.append(idx)

    # We delete empty lists.
    for empty_list_idx in reversed(sorted(empty_lists_to_pop_idx)):            
        lists.pop(empty_list_idx)

    clusters = []   

    while len(lists) > 0:
        
        most_confiable_object_idx, most_confiable_object_list_idx = get_most_confiable_object_idx(lists)
        most_confiable_object_list = lists[most_confiable_object_list_idx]
        most_confiable_object_list_bboxes, most_confiable_object_list_classes, most_confiable_object_list_confidences = most_confiable_object_list

        # We need to create a filter to exclude the selected object from lists.
        filter_idx = most_confiable_object_list_classes.shape[0]*[True]
        filter_idx[most_confiable_object_idx] = False

        # We get the most confiable object information and exclude it from the lists.
        most_confiable_object_bbox = most_confiable_object_list_bboxes[most_confiable_object_idx,:]
        most_confiable_object_list_bboxes = most_confiable_object_list_bboxes[filter_idx]
        most_confiable_object_class = most_confiable_object_list_classes[most_confiable_object_idx]
        most_confiable_object_list_classes = most_confiable_object_list_classes[filter_idx]
        most_confiable_object_confidence = most_confiable_object_list_confidences[most_confiable_object_idx]
        most_confiable_object_list_confidences = most_confiable_object_list_confidences[filter_idx]
        assert most_confiable_object_list_bboxes.shape[0] == most_confiable_object_list_classes.shape[0] == most_confiable_object_list_confidences.shape[0]


        # We update the list.
        lists[most_confiable_object_list_idx] = [most_confiable_object_list_bboxes, most_confiable_object_list_classes, most_confiable_object_list_confidences]

        # We generate the most confiable object tuple.
        most_confiable_object = (most_confiable_object_bbox, most_confiable_object_class, most_confiable_object_confidence)

        # We initialize the cluster.
        cluster_bboxes = [most_confiable_object_bbox]
        cluster_classes = [most_confiable_object_class]
        cluster_confidences = [most_confiable_object_confidence]

        empty_lists_to_pop_idx = []
        # Look for that object in other lists.

        for list_idx in range(len(lists)):
            if list_idx != most_confiable_object_list_idx:

                current_list = lists[list_idx]
                # We get the nearest object to most_confiable_object in current_list with a given IoU treshold.
                same_object_from_current_list_idx = get_same_object(most_confiable_object, current_list, threshold)
                # if not None, we insert it in cluster and delete from the list.

                if not same_object_from_current_list_idx is None:
                    # We get the object data.
                    current_list_bboxes, current_list_classes, current_list_conficendes = current_list

                    #We add data to the cluster.
                    cluster_bboxes.append(current_list_bboxes[same_object_from_current_list_idx])
                    cluster_classes.append(current_list_classes[same_object_from_current_list_idx])
                    cluster_confidences.append(current_list_conficendes[same_object_from_current_list_idx])

                    # We create a filter list to delete the selected object.
                    current_list_filter_idx = current_list_classes.shape[0]*[True]
                    current_list_filter_idx[same_object_from_current_list_idx] = False

                    #We update the current list data.
                    current_list_bboxes = current_list_bboxes[current_list_filter_idx]
                    current_list_classes = current_list_classes[current_list_filter_idx]
                    current_list_conficendes = current_list_conficendes[current_list_filter_idx]
                    

                    # if current list is not empty, we save the emaining data. If it's empty, we add to the list of lists to deleted.
                    if current_list_classes.shape[0] != 0:
                        current_list = current_list_bboxes, current_list_classes, current_list_conficendes
                        lists[list_idx] = current_list
                    else:
                        empty_lists_to_pop_idx.append(list_idx)

        cluster = (np.array(cluster_bboxes), np.array(cluster_classes), np.array(cluster_confidences))
        clusters.append(cluster)

        if most_confiable_object_list_classes.shape[0] == 0:
            empty_lists_to_pop_idx.append(most_confiable_object_list_idx)

        for empty_list_idx in reversed(sorted(empty_lists_to_pop_idx)):            
            lists.pop(empty_list_idx)

    return clusters