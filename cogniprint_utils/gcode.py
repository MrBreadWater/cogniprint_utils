from pygcode import Line
from pygcode.exceptions import GCodeWordStrError
from multiprocessing.pool import ThreadPool
import numpy as np

def read_line(line_text: str):
    '''Parse a single line of GCODE using pygcode.Line

    Args:
        line: A single line of GCODE

    Returns:
        A Line object, or None
    '''
    try:
        line = Line(line_text)
        return line
    except GCodeWordStrError:
        print("Warning: Unknown line!")

def get_points_from_line(gcode_line: Line, x: float, y: float, z: float):
    '''Get the XYZ coordinates from a pygcode.Line object

    Args:
        gcode_line: A pygcode.Line object
        x: float, current x position
        y: float, current y position
        z: float, current y position

    Returns:
        dict
    '''
    if not gcode_line:
        return None

    gcodes = next(iter(gcode_line.block.gcodes), None)

    if not gcodes:
        return None

    line_dict = gcodes.get_param_dict()

    if not line_dict:
        return None

    xyz = [line_dict.get(key, default) for key, default in zip(["X","Y","Z"], [x,y,z])]

    return xyz

def get_points_from_file(file : str ='/home/michael/Downloads/cat.gcode'):
    '''Get all points the printer head moves to from a given GCODE file

    Args:
        file: the path to the file

    Returns:
        A numpy.ndarray containing the set of extracted points
    '''
    with open(file, 'r') as fh:
        lines = fh.readlines()
        import sys
        sys.setrecursionlimit(len(lines))
        p = ThreadPool(16)
        gcode_lines = p.map(read_line, lines)
        #print("completed!")

    x = 0
    y = 0
    z = 0.15

    points = []

    for gcode_line in gcode_lines:
        xyz = get_points_from_line(gcode_line, x, y, z)
        if xyz:
            points.append(xyz)
            x, y, z = xyz

    points = np.array(points)

    return points
