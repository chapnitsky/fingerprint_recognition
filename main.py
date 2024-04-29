import cv2
import os
import numpy as np
from FingerprintAnalysisTemplate import ConcreteFingerprintAnalysis, client_code
from main_chains import BestRegionHandler, EnhancementHandler, MinutiaeHandler, NormalizationHandler, OrientationHandler, SegmentationHandler, SkeletonizationHandler
from main_full import count_fingerprint_ridges

def run_steps(image_path):
    block_size = 16
    image = cv2.imread(image_path, 0)

    best_region_handler = BestRegionHandler()
    minutiae_handler = MinutiaeHandler(best_region_handler)
    skeletonization_handler = SkeletonizationHandler(minutiae_handler)
    enhancement_handler = EnhancementHandler(skeletonization_handler)
    orientation_handler = OrientationHandler(enhancement_handler)
    segmentation_handler = SegmentationHandler(orientation_handler)
    normalization_handler = NormalizationHandler(segmentation_handler)

    # Our Design Patterns Implementations
    chains_output_images = normalization_handler.process((image, block_size))
    template_output_images = client_code(ConcreteFingerprintAnalysis(), image)

    # Original Implementation
    assaf_output_image = count_fingerprint_ridges(image)

    # Remove black images
    chains_output_images.pop(1)
    chains_output_images.pop(-3)

    # Images to compare
    chains_images = chains_output_images[:3][::-1]
    template_images = template_output_images[-3:]

    labels = ['gabor', 'skeleton', 'result']

    for name, original, chain, template in zip (labels, assaf_output_image, chains_images, template_images):
        canvas_height = max(original.shape[0], chain.shape[0], template.shape[0]) + 100
        canvas_width = original.shape[1] + chain.shape[1] + template.shape[1]

        if name == 'result':
            canvas = 255 * np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)
        else:
            canvas = 255 * np.ones((canvas_height, canvas_width), dtype=np.uint8)

        canvas[0:original.shape[0], 0:original.shape[1]] = original
        canvas[0:chain.shape[0], original.shape[1]:original.shape[1]+chain.shape[1]] = chain
        canvas[0:template.shape[0], original.shape[1]+chain.shape[1]:canvas_width] = template

        cv2.putText(canvas, name, (700, 600), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, 'Original', (150, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'Chains', (670, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'Template', (1200, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow(name, canvas)
        cv2.waitKey()


    

if __name__ == "__main__":
    path = os.getcwd() + '/all_png_files'
    img_name = "M89_f0115_03.png"
    img_path = f'{path}/{img_name}'
    run_steps(img_path)