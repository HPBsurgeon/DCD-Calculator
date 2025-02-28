# **Development and Validation of a Machine Learning Model to Reduce Futile Donation after Circulatory Death Procurements in Liver Transplantation: A National Multi-Center Study of 2,211 Potential Donors**

## **Authors**
Rintaro Yanagawa1, Kazuhiro Iwadoh2, Toshihiro Nakayama3, Daniel J Firl4, Chase J Wehrle5, Yuki Bekki6, Daiki Soma7, Jiro Kusakabe5, Yuzuru Sambommatsu8, Yutaka Endo6, Kliment K Bozhilov3, Jenny H Pan3, Masaru Kubota9, Koji Tomiyama6, Masato Fujiki5, Magdy Attia10, Marc L Melcher3, Kazunari Sasaki3

## **Affiliations**
1. Faculty of Medicine, Kyoto University, Kyoto, Japan.  
2. Department of Transplant Surgery, Mita Hospital, International University of Health and Welfare, Tokyo, Japan.  
3. Division of Abdominal Transplantation, Department of Surgery, Stanford University Medical Center, Stanford, California, USA.  
4. Department of Surgery, Duke University School of Medicine, Durham, North Carolina, USA.  
5. Department of Transplantation, Cleveland Clinic, Cleveland, Ohio, USA.  
6. Division of Hepatobiliary and Pancreatic Surgery, Department of Surgery, University of Rochester Medical Center, Rochester, New York, USA.  
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
Background: Liver transplantation using Donation after Circulatory Death (DCD) donors continues to grow in popularity, easing the organ shortage. However, there remain high rates of unsuccessful organ procurements, aka "futile procurements," most commonly due to a donor not expiring within an acceptable time period after withdrawal of care. Futile procurements pose significant financial and workload challenges. This study aimed to develop and validate a machine learning-based model to predict donor expiration and reduce futile procurement.

Methods: A retrospective dataset of 1,616 donors was used to develop a prediction model employing the Light Gradient Boosting Machine (LGBM). The model included neurological, biochemical, respiratory, and circulatory parameters as predictors and was validated retrospectively with 398 donors and prospectively with 207 donors across six U.S. transplant centers. Model performance was evaluated using the area under the curve (AUC), accuracy, futile procurement rate and missed opportunity rate. Performance was compared against established risk stratification models (DCD-N score, Colorado Calculator) and surgeon predictions.

Findings: Of the 2,211 DCD donors in this study, 1,260 expired. 903 (71·7%) expired within 30 minutes post-extubation. The LGBM model achieved AUCs of 0·83, 0·80, and 0·81 for predicting donor expiration at 30, 45, and 60 minutes post-extubation, respectively. This performance was maintained in both retrospective (AUCs: 0·83, 0·82, 0·80) and prospective (AUCs: 0·83, 0·81, 0·81) validation cohorts. The model reduced futile procurement rates compared to surgeons (0·08 vs. 0·20) and maintained superior accuracy in cases of poor inter-surgeon agreement (0·08 vs. 0·28). The missed opportunity rate was similar to that of surgeons (0·17 vs. 0·16).

Interpretation: This study demonstrates that an LGBM model can enhance predictive accuracy, reduce futile procurements, and outperform surgeons and traditional methods. Further improvements are needed to decrease missed opportunities and strengthen overall model accuracy.

Funding: None 

## **Software Implementation**
Source code used to creat prediction model and validation in the paper is stored in the main repository folder. 

### **Data Availability**
The dataset used in this research is not publicly available due to regulatory restrictions. The data contains sensitive patient information, making it subject to legal and ethical constraints that prevent its public release.

---

**Disclaimer:** This repository is maintained for research and academic purposes only. The authors and institutions bear no responsibility for unintended usage.
