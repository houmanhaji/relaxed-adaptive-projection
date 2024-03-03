import numpy as np


def get_enumeration(num_variables):
    enumeration = []
    for i in range(pow(2,num_variables)):
        combo = np.zeros(num_variables)
        # can we use np.argmin instead
        for j in range(num_variables):
            combo[j] = (int)(i / pow(2, num_variables - j - 1))
            if (combo[j]):
                i = i - pow(2, num_variables - j - 1)
        enumeration.append(combo)
    enumeration = np.asarray(enumeration)
    enumeration = 2 * enumeration - np.ones(enumeration.shape)
    return enumeration


def closest_point_on_line(line_point, line_direction, external_point):
    # Convert inputs to NumPy arrays for easier calculations
    line_point = np.array(line_point)
    line_direction = np.array(line_direction)
    external_point = np.array(external_point)

    # Calculate the direction vector squared magnitude
    v_magnitude_squared = np.dot(line_direction, line_direction)

    # Calculate the parameter t
    t = np.dot(line_direction, external_point - line_point) / v_magnitude_squared

    # Calculate the closest point on the line
    closest_point = line_point + t * line_direction

    return closest_point

