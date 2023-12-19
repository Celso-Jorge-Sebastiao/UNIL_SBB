# UNIL_SBB - Machine learning master's projet

## 1. Introduction 
 

The objective of this project was to classify french sentences using the CEDR evaluation. Sentences are assigned classes ranging from A1 (basic) to C2 (advanced). The associated skills for each level are detailed in the image below:
 
<img src="https://github.com/Celso-Jorge-Sebastiao/UNIL_SBB/assets/148785564/bf3b0a97-09bd-4c0a-8e44-eb2cd4598c7a" alt="image" width="1000"/>
 
The classification algorithm employed for this task was based on machine learning. We explored various approaches, including neural networks, word embedding, and different classifiers. In this GitHub repository, you will find all the data our group utilized for the assignment and our participation in the Kaggle competition.
 
Towards the end of this introduction, you will find an explanation of the assignment presented in video format on YouTube.

## 2. Progress overtime 

<img src="https://github.com/Celso-Jorge-Sebastiao/UNIL_SBB/assets/82185439/cbe41261-4476-4bb7-9c4b-4bc5fb45fe8e" alt="image" height = "300" width="1000"/>

The initial attempts involved applying a neural network algorithm. We began by dividing the sentences in our training dataset into a bag-of-words vector. To handle the labels, One-Hot Encoding was employed, transforming the labels into a binary vector with six dimensions. The selection of twenty stopwords was based on the cumulative sum of different columns from our bag-of-words. Various combinations of learning rates and hidden layers were experimented with. The initial score obtained was 0.468, which, although suboptimal, exceeded the default rate.

Subsequently, we explored the use of spaCy, an open-source library for advanced Natural Language Processing (NLP). SpaCy provides access to over 500 French stopwords stored in its library. While achieving a high score for the training test, it was evident that overfitting played a significant role. The final score mirrored that of the Neural Network Classifier.

In a further step, we turned to BERT, specifically the multi-language BERT, to embed our training dataset. Various classifiers were employed, resulting in an increased precision of 0.505 for the unlabelled dataset.

Lastly, CamemBERT, specialized in the French language, was leveraged to attain the final score. Alongside this, new classifiers were introduced. The initial result yielded a score of 0.564. After adjusting parameters and incorporating different classifiers, the final score improved to 0.569.

### Mentions to the following unsucessful attempts 

Data Augmentation:
One of our attempts involved data augmentation, where we sought to enrich our training data by leveraging ChatGPT to classify the overall difficulty levels of selected Jules Verne books. Subsequently, we downloaded copyright-free books, segmented the sentences, and appended the predicted levels from ChatGPT. While the training precision exhibited promising results, the precision on unlabelled data dropped to 0.50. We hypothesize that the labels provided by ChatGPT may not have been optimal, and assigning the same label to all sentences could be an oversimplification. Nevertheless, this attempt highlighted the potential benefits of data augmentation. The books and their associated levels are stored in CSV format within the "Livres" folder.

Hierarchical Prediction:
In our final attempt, we aimed to create four distinct predictions. The initial stage involved predicting the category level of sentences, distinguishing between A, B, and C while excluding the "1" and "2" parts. Subsequently, for each category, we built models to predict whether the level was "1" or "2." The "ABC" model was trained using a single feature—the number of words in the sentence. Additional attempts to include the number of commas resulted in decreased accuracy. The attempt was not fully completed as the precision of the "ABC" model multiplied by the precision of the "1/2" models did not appear promising. It is noteworthy that the "1/2" model exhibited notably high accuracy.

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

It's insightful to recognize that the model faces challenges in predicting correct labels, but the majority are accurate, with misclassifications typically being one class higher or lower than the true label. Addressing this issue could potentially involve augmenting the dataset with more diverse examples or incorporating additional features that differentiate between neighboring classes. These strategies may help enhance the model's ability to make more precise predictions and mitigate misclassifications. Experimenting with different features and collecting a more extensive dataset could contribute to refining the model's performance.

<img src="https://github.com/Celso-Jorge-Sebastiao/UNIL_SBB/assets/82185439/1bd94279-010c-46f4-9908-8d9e06ffee97" width="500">

## 5. User Interface 

The culmination of our work is an integrated algorithm presented through a Streamlit interface. This user-friendly tool aids in classifying PDFs containing French sentences. Here's how it works:

- Download PDFs:
  - Users can download copyright-free books in PDF format from https://bibliothequenumerique.tv5monde.com/liste/livres or any other preferred platform.
- Upload PDF:
  - Copy the path of the downloaded PDF file.
  - Paste the path into the provided input field on the Streamlit interface.
- Classification Process:
  - The application utilizes our developed algorithm to classify the PDF, predicting the overall language proficiency required to read the book. The algorithm is capable of splitting PDF sentences for analysis. To extract sentences from PDFs, we employed the "pdftosentence.py" code.
- Output:
  - Users receive the classification results, gaining insights into the global language proficiency needed for the given PDF.

This Streamlit interface enhances user accessibility and simplifies the classification process, making it convenient for users to assess the language proficiency associated with their PDF content.

<img src="https://github.com/Celso-Jorge-Sebastiao/UNIL_SBB/assets/82185439/c1173d75-0b6d-4636-b6a1-dada3b32e2bb" width="800">

*Note that the different steps may take a while to run

## 6. Youtube Presentation

[![Watch the video](https://img.youtube.com/vi/GM7pIDpK_Pg/maxresdefault.jpg)](https://www.youtube.com/watch?v=GM7pIDpK_Pg)

## 7. Paths

- CamemBERT (final code) : camembert_(final_code).py
- Natural Network : unsuccessful attempts/neural_network.py 
- SpaCy Newtork : unsuccessful attempts/spacy.py
- "ABC 1/2" model : unsuccessful attempts/testing_classification_embedding_bert.py
- User Interface Code : Streamlit_SBB.py
- Extract sentences from pdf : pdftosentence.py

## 8. Credits 

ChatGPT version 3.5 was used as a programming assistant and for debugging errors. 
All codes were written in the google colab platform.
The datasets provided were given by the assistants and the teacher. 

## 9. Context 

This assignment was collaboratively undertaken by Lisa Chauvet and Celso Jorge Sebastiao for Professor Michalis Vlachos' class on Data Mining and Machine Learning at the University of Lausanne (UNIL). The completion deadline for this assignment was set for December 20, 2023.

<img src="https://github.com/Celso-Jorge-Sebastiao/UNIL_SBB/assets/148785564/c8cb4c43-f77f-47f0-9569-39cc3429a300" alt="image" width="200"/>
