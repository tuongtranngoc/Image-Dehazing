import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm


def gen_lanes(image:np.ndarray):
    H, W, __ = image.shape
    
    W_ratio = 7
    H_ratio = 4
    
    xstart = np.random.randint(2, W_ratio-2)
    ystart = np.random.randint(2, H_ratio-1)
    
    _points = [
        [(W//W_ratio)*(xstart+1), (H//H_ratio)],
        [(W//W_ratio)*(xstart+2), (H//H_ratio)],
        [(W//W_ratio)*(xstart+3), (H//H_ratio)*(ystart+1)],
        [(W//W_ratio)*xstart, (H//H_ratio)*(ystart+1)],
    ]
    points = np.array(_points)
    points = points.reshape((-1, 1, 2))
    
    alpha = np.random.uniform(0.2, 0.5)
    thickness1 = 2
    thickness2 = 10
    num_points = 50
    isClosed = True
    color1 = (255, 0, 0)
    color2=(0, 100, 255)
    
    image = cv2.polylines(image, [points], isClosed, color1, thickness1)
    image = cv2.polylines(image, [points[-2:]], isClosed, color2, thickness2)
    
    delta = np.random.randint(-50, 50)
    width_offset1 = np.random.randint(-200, 200)
    if width_offset1 > 0:
        width_offset2 = width_offset1 + np.random.randint(200, 400)
    else:
        width_offset2 = width_offset1 - np.random.randint(200, 400)
    
    overlay = image.copy()
    cv2.fillPoly(overlay, [points], color=color1)
    
    # Define start point, end point of curve lines
    start_point1 = [_points[2][0] + delta, _points[2][1] + delta]
    end_point1 = [_points[1][0], _points[1][1]]
    
    start_point2 = [_points[3][0] + delta, _points[3][1] + delta]
    end_point2 = [_points[1][0], _points[1][1]]

    # Set strage lines with 2 points
    if np.random.uniform() > 0.5:
        num_points = 2
        width_offset2 = np.random.randint(100, 350)
        width_offset1 = -width_offset2
        
        image = draw_curve_lines(image, start_point1, end_point1, width_offset=width_offset1, num_points=num_points)
        image = draw_curve_lines(image, start_point2, end_point2, width_offset=width_offset2, num_points=num_points)
        
    else:     
        image = draw_curve_lines(image, start_point1, end_point1, width_offset=width_offset2, num_points=num_points)
        image = draw_curve_lines(image, start_point2, end_point2, width_offset=width_offset1, num_points=num_points)

    image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    return image_new


def draw_curve_lines(image, start, end, width_offset, num_points):
    line_thickness = 10
    color = (0, 100, 255)
    
    def create_parabolic_curve(center, start_y, end_y, width_offset, num_points):
        points = []
        for y in np.linspace(start_y, end_y, num_points):
            x_offset = width_offset * ((y - start_y) / (end_y - start_y))**2
            points.append((int(center[0] + x_offset), int(y)))
        return points
    
    center_x = start[0]
    start_y = start[1]
    end_y = end[1]
    
    curve_points = create_parabolic_curve((center_x, start_y), start_y, end_y, width_offset, num_points)
    image = cv2.polylines(image, [np.array(curve_points)], isClosed=False, color=color, thickness=line_thickness)
    return image


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_examples', type=str, default=100, help="Number of examples")
    parser.add_argument('--input_folder', help="Path to input folder")
    parser.add_argument('--output_folder', help="Path to output folder")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = cli()
    os.makedirs(args.output_folder, exist_ok=True)
    for img_path in tqdm(glob.glob(os.path.join(args.input_folder, '*'))[900:900+args.num_examples]):
        img = cv2.imread(img_path)
        img = gen_lanes(img)
        cv2.imwrite(os.path.join(args.output_folder, os.path.basename(img_path)), img)
    
    