# ISICClassifier

### Author: Dhruv Joshi
### Date: December 2019

In this project, we create a model that will use data such as an image, age, and sex of a patient to diagnose the patient's skin anomaly as benign or malignant.

The model created achieved a specificity of 0.96 and a sensitivity of 0.52.

This means that the model was able to only correctly diagnose 52% of the people who had the disease, this is near the expected outcome of guessing. However, the model manages to correctly diagnose 96% of the people who did not have the disease. 

This suggests that the model is very effective at ruling in patients. That is, if the model states malignancy, then the patient should be ruled into having a malignant mole. This allows for the model to be effecient to rule in users who should be seen by a doctor for further diagnoses and medical attention.

To achieve these results, a VGG16 pretrained model without the last 3 layers is used. These results are then input into an MLP to end up with a final diagnoses for the given patients image. 

A model that included the Age and Sex information of the patient performed worse than the image only model. This architecture combined the output vector of the VGG16 model without the top 3 layers with 10 output neurons of an MLP as an input to another MLP that output the diagnoses for the patient at hand.

In all, we are able to conclude that the model perfomrs well for correctly diagnosing people who do not have the disease. To improve results, we would need to increase data, or better understand the features that go into correct diagnoses for the disease (such as features relating to the ABCDE technique of preliminary analysis of moles) and use these methods to better provide the model with input features. 

Further recommendations for improving the model include, fine tuning the transfer learning model, improving the MLP for mixed data, trying different pretrained models, change of loss function to reflect the importance of a mis diagnoses (false negative being significantly worse than a false positive), and improving the image preprocessing to provide mole masks, remove hair, etc.


