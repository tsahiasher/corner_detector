import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

def parse_yolo_keypoint_line(line: str) -> Optional[List[List[float]]]:
    """Parses a YOLO format keypoint annotation line.
    
    Expected format: class_id cx cy w h k1x k1y k1v k2x k2y k2v k3x k3y k3v k4x k4y k4v
    
    Args:
        line (str): A single line from a YOLO txt file.
        
    Returns:
        Optional[List[List[float]]]: A list of 4 keypoints (x, y) normalized [0,1],
        in the fixed order [top-left, top-right, bottom-right, bottom-left], 
        if parsing is successful. None otherwise.
        
    Raises:
        ValueError: If line cannot be parsed as a float list.
    """
    try:
        parts = list(map(float, line.strip().split()))
        if len(parts) >= 17:
            keypoints = []
            for i in range(4):
                x = parts[5 + i*3]
                y = parts[6 + i*3]
                keypoints.append([x, y])
            
            # Deterministically sort to ensure TL, TR, BR, BL order
            import math
            cx = sum(pt[0] for pt in keypoints) / 4.0
            cy = sum(pt[1] for pt in keypoints) / 4.0
            def angle_from_center(pt):
                return math.atan2(pt[1] - cy, pt[0] - cx)
                
            keypoints = sorted(keypoints, key=angle_from_center)
            return keypoints
    except ValueError as e:
        logger.error(f"Failed to parse YOLO line: {line}. Error: {e}")
    return None
