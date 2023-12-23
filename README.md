# Retinal-Vessel-Extraction-by-Matched-Filter-with-First-Order-Derivative-of-Gaussian
This is an advanced image processing project focused on enhancing medical imaging analysis through precise retinal vessel detection.



## Motivation & Contributions:
The motivation stems from the necessity to improve the accuracy of retinal vessel extraction, which is a key step in diagnosing and managing various ocular diseases. Zhang et al.'s "Retinal vessel extraction by matched filter with first-order derivative of Gaussian" proposes the MFFDOG technique, prioritizing simplicity and effectiveness to improve vessel identification, especially in pathological cases [1], and Di Li et al.'s "RESIDUAL U-NET FOR RETINAL VESSEL SEGMENTATION" presents the modified residual U-Net (ResU-Net), a deep learning model that excels at capturing complex features for accurate segmentation [2].


## Main Approaches:
MF-FDOG utilizes a combination of the traditional matched filter (MF) and the first-order derivative of Gaussian (FDOG). It thresholds the retinal image's response to the MF, with the threshold adjusted by the response to the FDOG. In addition, the model's preprocessing and
postprocessing stages crucially refine the results. This results in improved discrimination between true vessel structures and other similar patterns [1]. The ResU-Net employs a deep convolutional network structure that integrates residual learning to enable the training of deeper models. It uses batch normalization before activation units and dropout layers to mitigate overfitting, aiming to achieve higher sensitivity and specificity in vessel detection [2].

## Critique of Presentation
In [1], the method is straightforward and well-presented, except for some preprocessing and postprocessing methods. The paper could benefit from a broader comparison with other simple vessel extraction techniques to highlight its advantages in different scenarios. Additionally, the paper used a logical OR, which is not adequate enough to remove unwanted structures, and noisy patterns from either scale may persist in the resultant vessel map. Furthermore, the paper could benefit from improving the identification and connectivity of vessels. If isolated vessels are connected to the correct objects, both sensitivity and accuracy could be significantly enhanced.


In [2], the authors clearly present their methods and results, with substantial comparative analysis
against state-of-the-art techniques. However, the complexity of the deep learning model may not
be readily accessible to practitioners without a background in the field. Additionally, it would be
beneficial if the authors applied their strategy to other CNNs, which could lead to improvements
in model adaptability or efficiency.



## Comparison
Both papers address the challenge of retinal vessel segmentation with innovative approaches. The ResU-Net shows superior performance due to its deep learning framework, which can model complex vessel structures. Meanwhile, the MF-FDOG is notable for its computational efficiency and simplicity, making it suitable for real-time applications. The results from ResU-Net are better due to the deep network's ability to learn from a large amount of data and extract nuanced features that simple filters cannot.


## References:
**1 - Zhang, B., Zhang, L., Zhang, L., & Karray, F. (2010). Retinal vessel extraction by matched filter with first-order derivative of Gaussian. Computers in biology and medicine, 40(4), 438-445.

2 - Li, D., Dharmawan, D. A., Ng, B. P., & Rahardja, S. (2019, September). Residual u-net for retinal vessel segmentation. In 2019 IEEE International Conference on Image Processing (ICIP) (pp. 1425-1429). IEEE.**



## 
## **Implementeation**

Our reimplementation has mainly achieved the results of the original paper with a high degree of accuracy. Nonetheless, we encountered some issues with noise. While we attempted to mitigate these noise issues, we were careful not to deviate significantly from the core principles of the image processing methods stated in the paper. Here's a concise overview of the project, including the coding aspect:
In the initial phase, we developed a class named MFR, comprising several functions such as filter_kernel_mf_fdog, fdog_filter_kernel, gaussian_matched_filter_kernel, createMatchedFilterBank, and applyFilters.


Detailed explanations of these functions are provided in the following sections.
The MFR class is structured to offer capabilities for generating both MF Gaussian and first-order derivative of Gaussian (FDOG) filter kernels. It also enables the creation of a collection of rotated kernels suited for different orientations (in 8 different directions), along with the functionality to apply these diverse filters to images.

Detailed explanation for each method is as follows:

**filter_kernel_mf_fdog**: This method generates either a matched filter or an FDOG kernel, depending on the specified parameters.

**fdog_filter_kernel**: This function acts as a convenient interface to _filter_kernel_mf_fdog, setting the mf parameter to False. It's designed for crafting an FDOG kernel, primarily used in edge detection tasks.

**gaussian_matched_filter_kernel**: By invoking _filter_kernel_mf_fdog with mf set to True, this method creates a Gaussian kernel. The resulting kernel is instrumental in both smoothing and accentuating vessel-like structures within images.

**createMatchedFilterBank**: This function assembles an array of rotated kernels, enhancing the system's capability to detect vessels across a range of orientations. It significantly boosts the algorithm's proficiency in recognizing vessels, irrespective of their orientation in the imagery.

**applyFilters**: Deploying a collection of filters on an image, this method amalgamates the responses from each filter. The outcome is a detailed representation of the detected vessels in the image.


**is_coordinate_within_bounds**: Checks whether a specified pixel coordinate falls within the image's boundaries, ensuring that all image manipulations are confined to its actual dimensions.

**label_connected_region**: Employs a region-growing technique to assign a uniform label to interconnected pixels that form a coherent region, with a limit on the maximum size of the region. This function utilizes an 8-connected neighbor approach, considering pixels that are directly adjacent either horizontally, vertically, or diagonally.

**load_and_preprocess_image**: This function retrieves an image and its corresponding mask from provided file paths and processes them. It converts the mask to gray scale and inverts the green channel of the image. Inverting the green channel is a standard practice in retinal image analysis, often yielding improved contrast for identifying blood vessels.


Subsequently, we developed a function named 'generator,' which acts as a comprehensive pipeline for processing and segmenting vessels in images.

**Generator** Function: Tailored for image processing, this function leverages a sequence of filters and image manipulation techniques to accentuate and identify key structures within the image, facilitating detailed analysis.

**Load and Preprocess**: The function starts by loading and preprocessing an image along with its mask, preparing it for further processing steps.

**Gaussian and FDOG Filters**: It initializes the MFR class and generates Gaussian and FDOG filters, which are then used to create a filter bank. These filters are applied to the green channel of the image to highlight features like vessels.

**Smoothing and Normalization**: Post-filter application, the image undergoes a smoothing process, and the response from the FDOG filter is normalized. This step is followed by the application of a threshold to the Gaussian response, further refining and highlighting key image features.

In our experiments, we tested various algorithms, finding that the Laplacian method yielded the most favorable results. When we implemented the Canny algorithm, we observed that it failed to eliminate edges effectively. Additionally, the application of the Hough Circle Transform did not produce results consistent with those documented in the paper. The outputs of these methods are illustrated in Fig 2.



**Laplacian Filtering**: The Laplacian filter is applied to the mask to focus on the region of interest within the image, helping in segmenting structures. The application of the Laplacian filter has highlighted the edges within the retinal image by enhancing the contrast of the blood vessels against the background. The Laplacian filter's role in this process is to accentuate areas of rapid intensity change, which typically correspond to the edges of vessels.

![image](https://github.com/SoroushGhorbanimehr/Retinal-Vessel-Extraction-by-Matched-Filter-with-First-Order-Derivative-of-Gaussian/assets/62909269/ddf39657-335a-48e8-9eda-a088f79d5924)


**Canny Edge Detection**: The Canny edge detection algorithm is applied to get better result but it has some advantage and disadvantage. It makes the vessels near the edge more visible, however, the paper results does not have the edge.
Hough Circle Transform: The Hough Circle Transform is utilized to find vessels. It makes the edge more thick but the result is not much better than others. 

**Segment Removal and remove small segments**: The function also includes a process to remove small segments that may be artifacts or noise, using connected component labeling and filtering out segments below a certain size. This function used to remove some noises pixel to make the image more clear. We filtered out segments smaller than 10 pixels.

**Return Statement**: The function returns the processed image with enhanced and segmented features ready for analysis, alongside the Gaussian response which may be used for further processing or analysis steps.



Subsequently, we subjected the 40 images from the **DRIVE** dataset to the algorithm for processing.



In the paper, a key post-processing step involved merging both thick and thin vessels using a logical OR operation. We encountered a challenge where certain parameters (L1, sigma1, w1, c1 = 5, 1, 31, 2.3) highlighted the thin vessels, while a different set of parameters (L, sigma, w, c = 9, 1.5, 31, 2.3) as mentioned in the paper, made the thick vessels more prominent. To address this, we combined the results from these two parameter sets using np.logical_or, which led to improved outcomes.


The outcomes were quantified using a function named calculate_metrics, as shown in the following code snippet. Primarily, this function computes key metrics such as True Positive Rate (TPR), False Positive Rate (FPR), and accuracy. These metrics are crucial in evaluating the effectiveness of the segmentation algorithm. Generally, an algorithm is considered to perform well if it achieves a high TPR and accuracy, coupled with a low FPR. It should be noted that we calculated AUC in the code, but due to the page limitation, we did not present it here.



## **Results:**
We conducted experiments using different parameter combinations and determined that the parameters outlined in the MF-DOG method are well-matched for the DRIVE database. As depicted in Fig1, the combination of both thick and thin vessels led to slightly improved results.

<p align="center">

![image](https://github.com/SoroushGhorbanimehr/Retinal-Vessel-Extraction-by-Matched-Filter-with-First-Order-Derivative-of-Gaussian/assets/62909269/6520a0d8-f0fd-44c3-ad53-60f480d5a73a)
</p>


The outcomes obtained by incorporating the Laplacian, Canny, and Hough Filter Transform methods in conjunction with the primary approach are depicted in Fig2. This analysis illustrates the impact and effectiveness of these techniques on the performance of the main approach.

<p align="center">

![image](https://github.com/SoroushGhorbanimehr/Retinal-Vessel-Extraction-by-Matched-Filter-with-First-Order-Derivative-of-Gaussian/assets/62909269/3d0e09cb-5226-4027-af91-3b47bcfc20b3)
</p>




Fig3. illustrates a comparative analysis between MF-FDOG output and the Ground Truth. This visual representation provides an examination of the disparities and agreements between the results produced by the MF-DOG method and the actual Ground Truth, contributing insights into the accuracy and reliability of the employed approach.


<p align="center">

![image](https://github.com/SoroushGhorbanimehr/Retinal-Vessel-Extraction-by-Matched-Filter-with-First-Order-Derivative-of-Gaussian/assets/62909269/7dd09448-6ddc-43e7-be6d-1080989c7682)
</p>


The results from each method, encompassing MF-FDOG and the combination of MF-FDOG with Laplacian, Hough Transform, and Canny techniques, indicate that the use of Laplacian in combination with MF-FDOG outperforms the other methods, as shown in Table 1.



<p align="center">

![image](https://github.com/SoroushGhorbanimehr/Retinal-Vessel-Extraction-by-Matched-Filter-with-First-Order-Derivative-of-Gaussian/assets/62909269/1f95f809-e78f-40f7-9850-3c9aded6643f)
</p>




While re-implementing the paper, we encountered both positive and negative aspects, which can be outlined as follows:


Pros:
● The integration of two techniques (MF + FDOG) resulted in a modest enhancement of the final outcome.
● Employing a logical OR operation improved segmentation of both thin and thick vessels, yet the improvements were relatively minor.
● Since we added several preprocessing and postprocessing techniques, we successfully some removed noises and improved the clarity of the images, yielding superior results compared to the implementation that lacked these approaches.

Cons:
● Since this paper used traditional methods it has lower accuracy in comparison to newer methods such as deep learning.
● There are still several noises in the result.
● The implementation process demanded meticulous and occasionally intricate parameter tuning, leading to potential time consumption. Although we experimented with different parameter ranges, ultimately, the parameters specified in the paper were found to be more effective.


One of the hardest parts for us was achieving the paper's results, because the authors did not exactly mention their preprocessing and post-processing methods. We had to try several methods, and it was really time-consuming. This project required a significant amount of time to correctly implement and achieve the results that the paper achieved. In addition, we prepared some plots for the ROC and launched the project with several parameters, which we could not include in this report due to page limitations.








## Conclusion:

In this project, we undertook the re-implementation of the MF+FDOG method, an extension of the MF approach known as MF-FDOG, specifically designed for retinal blood vessel detection. The proposed MF+FDOG combines the original MF, a zero-mean Gaussian function, with the first-order derivative of Gaussian (FDOG). The parameter values of MF-FDOG were set according to the image properties as specified in the base paper. In summary, the MF-FDOG method demonstrated effective retinal blood vessel segmentation.
To further refine this method, we aimed to reduce the noise inherent in the MF-FDOG approach by eliminating connected pixels comprising less than 10 pixels. Additionally, by integrating this refined approach with the Laplacian filter, we successfully achieved an improved segmentation outcome. This combined method not only enhances the noise reduction capabilities but also contributes to an overall superior segmentation performance.


