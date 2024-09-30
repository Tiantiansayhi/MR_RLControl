import numpy as np
current_image_position = np.array([0,0])
desired_image_position = np.array([200,200])
print(np.linalg.norm(current_image_position - desired_image_position))
