# UNIL_SBB - Machine learning master's projet

<img src="https://github.com/Celso-Jorge-Sebastiao/UNIL_SBB/assets/148785564/bf3b0a97-09bd-4c0a-8e44-eb2cd4598c7a" alt="image" width="1000"/>



## 1. Introduction :

The goal of this work was to classify sentences using the CEDR evaluation. The classes assigned to the sentences range from A1 (basic) to C2 (advanced). The algorithm used to achieve the classification was based on machine learning, various attempts were made using neural networks, word embedding and different classifiers. 
In this github repository, you will find all the data our group used in order to do the assignment and compete in the kaggle's competition. 

At the end of this presentation, wou will find the explanation of this assignement in a youtube video format. 


## 2.Progress overtime :

The first attempts were based on a neural network algorithm. Firstly, we split the sentences from our training dataset into a bag of words vector. We used OneHotEncoding to split the labels that we wanted to predict into a 6 dimensions binary vector. The twenty stopwords used were chosen based on the sum of the different columns from our bag of words. We tried various numbers of learning rates and hidden layers. The inital score we got was XXX which was a bad result but still higher than the default rate. You can find the code in the XXX folder. 

Then, we tried to used spaCy which is an open-source library for advanced Natural Language Processing (NLP). Spacy offers the possibility to use 500+ french stopwords already stored in the library. We got a very good result for the training test but it was only due to overfitting. The final score was slightly better than the Neural Network Classifier. 

Furthermore, we tried to use Bert, multi-language Bert, to embed our training dataset. We  used different classifiers. The precision for the unlabelled dataset increased to nearly XXX.

Finally, CamemBert, which is specialized in french language was used to achieve the final score. Along with new classifiers presented below. 

Mentions to the following uncessful attempts : 
Data augmentation : We tried to add training data by asking ChatGPT to classify the global level of some Jules Vernes books. Then we downloaded the copyright free books, split the sentences and added the level provided by ChatGPT into all of them. 
The training precision was great but the unlabelled precision went down to 0.50 again. 
We believe that the labels predicted by ChaGPT were not that great and that giving the same label to all the sentences is also wrong. But this attempt showed that data augmentation might provide better precision.
The books and their level are stored in a csv format in the "Livres" folder. 

The last attempt was to create 4 different predictions 
The first part consists of prediction the category level of the sentences, it means that we only predict if the sentence is A, B or C and drop the "1" and "2" part. Then, for each category, we create a model that predicts if it's "1" or "2".
The "ABC" model is training using one unique feature which is the number of words in the sentence. We also tried to add the number or commas but the accuracy decreased. The attempt was not completed because the precision of the "ABC" model multiplied by the precision of the "1/2" models didnt seem promising. Noting that the "1/2" model showed very high accuracy. 

GIF

## 3. Classifiers

### Logistic Regression
-----------------------
Type: Supervised learning algorithm for classification.
Usage: Used when the dependent variable is binary (two classes).
How it works: It models the probability that each input belongs to a particular category and uses the logistic function to transform raw predictions into probabilities.

### K-Nearest Neighbors (KNN) Classifier
----------------------------------------
Type: Instance-based learning algorithm.
Usage: Used for both classification and regression tasks.
How it works: It classifies a data point based on how its neighbors are classified. The class of a data point is determined by majority voting of its k-nearest neighbors.

### Gaussian Naive Bayes (GaussianNB)
-------------------------------------
Type: Probabilistic algorithm, based on Bayes' theorem.
Usage: Commonly used for classification tasks, especially when dealing with text data.
How it works: Assumes that features are conditionally independent given the class and uses probability distributions (Gaussian in this case) to make predictions.

### Support Vector Classifier (SVC)
-----------------------------------
Type: Supervised learning algorithm for classification.
Usage: Effective for both binary and multi-class classification tasks.
How it works: Finds a hyperplane that best separates the data into classes by maximizing the margin between the classes.

### Random Forest Classifier
----------------------------
Type: Ensemble learning algorithm.
Usage: Effective for both classification and regression tasks.
How it works: Builds multiple decision trees during training and merges them together to get a more accurate and stable prediction.

### Decision Tree Classifier
----------------------------
Type: Supervised learning algorithm.
Usage: Used for both classification and regression tasks.
How it works: Divides the dataset into subsets based on the most significant attributes, forming a tree-like structure. Each leaf node represents a class label.

### Voting Classifier
---------------------
Type: Ensemble learning method.
Usage: Combines the predictions from multiple base classifiers and outputs the majority vote.
How it works: It can be hard or soft voting. In hard voting, each classifier votes for a class, and the majority class is chosen. In soft voting, the predicted class probabilities are averaged, and the class with the highest average is chosen.

### Stacking Classifier
-----------------------
Type: Meta-model or ensemble learning method.
Usage: Combines multiple base classifiers with a meta-classifier to improve predictive performance.
How it works: The base classifiers make predictions, and their outputs become inputs to a meta-classifier, which then makes the final prediction. This helps capture more complex relationships in the data.


### Performance Metrics by Classifier

| Metric   | Logistic Regression | K-Nearest Neighbors (KNN) Classifier | Gaussian Naive Bayes (GaussianNB) | Support Vector Classifier (SVC) | Random Forest Classifier | Decision Tree Classifier | Voting Classifier | Stacking Classifier |
|----------|----------------------|--------------------------------------|-----------------------------------|----------------------------------|--------------------------|--------------------------|-------------------|----------------------|
| Accuracy | 0.565                | 0.502                                | 0.483                             | 0.569                            | 0.520                    | 0.353                    | 0.566             | 0.571                |
| Precision| 0.565                | 0.502                                | 0.483                             | 0.569                            | 0.520                    | 0.353                    | 0.566             | 0.571                |
| Recall   | 0.565                | 0.502                                | 0.483                             | 0.569                            | 0.520                    | 0.353                    | 0.566             | 0.571                |
| F1 Score | 0.565                | 0.502                                | 0.483                             | 0.569                            | 0.520                    | 0.353                    | 0.566             | 0.571                |

The final classifier chosen was stacking classifier associated with the following ones :
- Logistic Regression (LR)
- Support Vector Classifier (SVC):
- and finally voting (LR + SVC, soft)

## 4. Confusion Matrix 

We can see that there is some problems to predict the correct labels but the majority are right and the misclassified are generally one class higher or lower than the true label. This can be corrected with more data in our dataset or maybe adding features that distinguish the neighbor classes. 

<img src="https://github.com/Celso-Jorge-Sebastiao/UNIL_SBB/assets/82185439/1bd94279-010c-46f4-9908-8d9e06ffee97" width="500">

## 5. User Interface 

The final algorithm was integrated into a streamlit interface to help the user classify a pdf containing french sentences. The user can download copyright free books in a pdf format in https://bibliothequenumerique.tv5monde.com/liste/livres or in any other prefered platform. Then he copies the path of the downloaded file and pastes it in the user interface. The application will classify the pdf and predict the global language proficiency needed to read the book. To split the pdf sentences, we used the code "Pdf_to_sentences.py".

<img src="https://github.com/Celso-Jorge-Sebastiao/UNIL_SBB/assets/82185439/c1173d75-0b6d-4636-b6a1-dada3b32e2bb" width="800">

## 6. Youtube Presentation

[![Watch the video](https://img.youtube.com/vi/GM7pIDpK_Pg/maxresdefault.jpg)](https://www.youtube.com/watch?v=GM7pIDpK_Pg)

## 7. Credits 

ChatGPT version 3.5 was used as a programming assistant and for debugging errors. 
All codes were written in the google colab platform.
The datasets provided were given by the assistants and the teacher. 

## 8. Context 

This assignment was made by Lisa Chauvet-Heinz and Celso Jorge Sebastiao for professor Michaelis Vlachos' UNIL class : Machine learning and supervised methods. 
The deadline was scheduled at the following date : 20.12.2023

<img src="https://github.com/Celso-Jorge-Sebastiao/UNIL_SBB/assets/148785564/c8cb4c43-f77f-47f0-9569-39cc3429a300" alt="image" width="200"/>
