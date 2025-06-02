# **Development and Validation of a Machine Learning Model to Reduce Futile Procurements in Donations After Circulatory Death in Liver Transplantation: A National Multi-Center Study**

## **Authors**
Rintaro Yanagawa1, Kazuhiro Iwadoh PhD2, Toshihiro Nakayama MD3, Daniel J Firl MD4, Chase J Wehrle MD5, Yuki Bekki PhD6, Daiki Soma MD7, Jiro Kusakabe MD5, Yuzuru Sambommatsu MD8, Yutaka Endo PhD 6, Kliment K Bozhilov MD3, Jenny H Pan MD3, Masaru Kubota PhD9, Koji Tomiyama PhD6, Masato Fujiki MD5, Magdy Attia MD10, Marc L Melcher PhD3, Kazunari Sasaki MD3

## **Affiliations**
1. Faculty of Medicine, Kyoto University, Kyoto, Japan.  
2. Department of Transplant Surgery, Mita Hospital, International University of Health and Welfare, Tokyo, Japan.  
3. Division of Abdominal Transplantation, Department of Surgery, Stanford University Medical Center, Stanford, California, USA.  
4. Department of Surgery, Duke University School of Medicine, Durham, North Carolina, USA.  
5. Department of Transplantation, Cleveland Clinic, Cleveland, Ohio, USA.  
6. Division of Transplant Surgery, Department of Surgery, University of Rochester Medical Center, Rochester, New York, USA.  
7. Department of Surgery, University of Florida College of Medicine, Gainesville, Florida, USA.  
8. Division of Transplant Surgery, Department of Surgery, Virginia Commonwealth University Health, Richmond, Virginia, USA.  
9. Division of Liver and Abdominal Transplantation, Department of Surgery, Columbia University Irving Medical Center, New York, New York, USA.  
10. Medical Affairs, TransMedics, Inc., Andover, Massachusetts, USA.  

## **Corresponding Author**
Kazunari Sasaki, M.D.  
Division of Abdominal Transplant, Department of Surgery, Stanford University Medical Center, Stanford, California, USA.  
Email: sasakik@stanford.edu  
Telephone: (650) 723-5454 (office)  

## **Abstract**
Background: The number of liver transplants using Donation after Circulatory Death (DCD) donors continues to increase, easing the organ shortage. However, there remains a high rate of attempted but subsequently aborted procurements, or "futile procurements," most commonly due to a donor not expiring within an acceptable time after extubation. Futile procurements pose significant financial and workload burdens. This study aimed to develop and validate a machine learning model to better predict donor expiration and reduce futile procurements.

Methods: A retrospective dataset of 1,616 donors was used to develop a prediction model employing the Light Gradient Boosting Machine (LGBM). The model included neurological, biochemical, respiratory, and circulatory parameters as predictors and was validated retrospectively with 398 donors and prospectively with 207 donors across six U.S. transplant centers. Model performance was evaluated using the area under the curve (AUC), accuracy, futile procurement rate, and missed opportunity rate. Performance was also compared to existing risk models (DCD-N score, Colorado Calculator) and surgeon predictions. 

Findings: Of the 2,221 DCD donors in this study, 1,260 donors expired. 927 donors expired within 30 minutes post-extubation. Cross-validation of the LGBM model achieved AUCs of 0·83, 0·80, and 0·81 for predicting donor expiration at 30, 45, and 60 minutes post-extubation, respectively. This performance was maintained in both retrospective (AUCs: 0·83, 0·82, 0·80) and prospective (AUCs: 0·83, 0·81, 0·81) validation cohorts. The model reduced futile procurement rates compared to surgeon predictions (8% vs. 20%) and maintained superior accuracy in cases of poor inter-surgeon agreement (8% vs. 28%). The missed opportunity rate was similar to that of surgeons (17% vs. 16%).

Interpretation: This study demonstrates that the LGBM model can enhance predictive accuracy, reduce futile procurements, and outperform surgeons and traditional methods. Further improvements are needed to decrease missed opportunities and strengthen overall model accuracy.
Funding: None 


## **Software Implementation**
Source code used to creat prediction model and validation in the paper is stored in the main repository folder. 

### **Data Availability**
The dataset used in this research is not publicly available due to regulatory restrictions. The data contains sensitive patient information, making it subject to legal and ethical constraints that prevent its public release.

---

**Disclaimer:** This repository is maintained for research and academic purposes only. The authors and institutions bear no responsibility for unintended usage.
