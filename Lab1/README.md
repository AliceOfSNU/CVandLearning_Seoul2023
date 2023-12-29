### Weakly supervised detection network
- WSDDN.py

    - uses pre-trained AlexNet's feature extraction layers
    - ROI's pre-extracted by selective search.
    - experiments with ROI_pooling/SPP

- SPP?
    
    - Spatial Pyramid Pooling
    - Max pool operation on each ROI, with different window size
    - concatenates results with different scale factors

___
a pytorch implementation of a simplified version of WSDNN from paper:

[1]
Bilen, Hakan, and Andrea Vedaldi.**"Weakly supervised deep detection networks."**  Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.


### class-wise mAP calculation
- multi-class detection scores

    - combined result of classification/detection heads for all region proposals give a per-image score

    - 0~1 score for each class, 1 means existence of that class in the image.

    - the score do not sum up to 1, because there may be multiple classes in one image

- implementation of mAP:
https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
from this site, interpolated AP for 20 points was used.
![Alt text](results/map_calc.png)

___
Part of the skeleton provided by CMU's Visual Learning course.
