const questions = [
     {
       question: "What is sklearn short for?",
       options: ["Scikit Learn", "SciPy Learn", "Scientific Learn", "Scikit Processing"],
       correctAnswer: 0,
       explanation: "Scikit-learn is a machine learning library for Python, that provides a range of tools for building and working with machine learning models."
     },
     {
       question: "Which of the following is NOT a type of supervised learning algorithm in Scikit-learn?",
       options: ["Decision Tree", "Clustering", "Random Forest", "Support Vector Machine"],
       correctAnswer: 1,
       explanation: "Clustering is an unsupervised learning algorithm in Scikit-learn. The other three options are supervised learning algorithms."
     },
     {
       question: "Which of the following is NOT a step in the typical machine learning workflow using Scikit-learn?",
       options: ["Preprocessing data", "Evaluating model performance", "Deploying the model", "Training the model"],
       correctAnswer: 2,
       explanation: "Deploying the model is not a step in the typical machine learning workflow using Scikit-learn. The other three options are important steps in the process."
     },
     {
       question: "Which of the following is NOT a type of kernel used in Support Vector Machines in Scikit-learn?",
       options: ["Linear", "Polynomial", "Sigmoid", "K-means"],
       correctAnswer: 3,
       explanation: "K-Means is not a type of kernel used in Support Vector Machines in Scikit-learn. The other three options are valid types of kernels."
     },
     {
       question: "Which of the following metrics can be used to evaluate a binary classification model in Scikit-learn?",
       options: ["Mean Squared Error", "F1 Score", "R Squared", "Adjusted R Squared"],
       correctAnswer: 1,
       explanation: "F1 Score is a common metric used to evaluate the performance of binary classification models in Scikit-learn. Mean Squared Error, R Squared, and Adjusted R Squared are not typically used for binary classification problems."
     },
     {
       question: "Which of the following is NOT a type of cross-validation technique available in Scikit-learn?",
       options: ["K-Fold Cross Validation", "Stratified K-Fold Cross Validation", "Leave-One-Out Cross Validation", "K-Nearest Neighbors Cross Validation"],
       correctAnswer: 3,
       explanation: "K-Nearest Neighbors Cross Validation is not a type of cross-validation technique available in Scikit-learn. The other three options are valid types of cross-validation techniques."
     },
     {
       question: "Which of the following is NOT a type of ensemble learning algorithm available in Scikit-learn?",
       options: ["Random Forest", "AdaBoost", "Gradient Boosting", "Logistic Regression"],
       correctAnswer: 3,
       explanation: "Logistic Regression is not an ensemble learning algorithm in Scikit-learn. The other three options are valid ensemble learning algorithms."
     },
     {
       question: "Which of the following is NOT a type of feature scaling technique available in Scikit-learn?",
       options: ["StandardScaler", "MinMaxScaler", "MaxAbsScaler", "Z-ScoreScaler"],
       correctAnswer: 3,
       explanation: "Z-ScoreScaler is not a type of feature scaling technique available in Scikit-learn. The other three options are valid types of feature scaling techniques."
     },
     {
       question: "Which of the following functions can be used to split a dataset into training and testing sets in Scikit-learn?",
       options: ["train_test_split", "split_dataset", "split_train_test", "create_train_test_set"],
       correctAnswer: 0,
       explanation: "train_test_split."
     },
     {
       question: "Which of the following Scikit-learn functions can be used to perform Principal Component Analysis (PCA)?",
       options: ["LinearRegression", "DecisionTreeClassifier", "LogisticRegression", "PCA"],
       correctAnswer: 3,
       explanation: "PCA is a dimensionality reduction technique that can be used to transform a dataset into a lower-dimensional space. The PCA function in Scikit-learn can be used to perform PCA on a dataset. LinearRegression, DecisionTreeClassifier, and LogisticRegression are not functions used for PCA."
     },
     {
       question: "Which Scikit-learn function is used to evaluate the performance of a regression model?",
       options: ["mean_absolute_error", "classification_report", "confusion_matrix", "accuracy_score"],
       correctAnswer: 0,
       explanation: "The mean_absolute_error function in Scikit-learn can be used to evaluate the performance of a regression model. The other three options are typically used to evaluate the performance of classification models."
     },
     {
       question: "Which Scikit-learn function can be used to handle missing values in a dataset?",
       options: ["Imputer", "StandardScaler", "MinMaxScaler", "OneHotEncoder"],
       correctAnswer: 0,
       explanation: "The Imputer function in Scikit-learn can be used to handle missing values in a dataset. StandardScaler, MinMaxScaler, and OneHotEncoder are used for feature scaling and encoding categorical features."
     },
     {
       question: "Which Scikit-learn function is used to tune the hyperparameters of a machine learning model?",
       options: ["GridSearchCV", "RandomForestClassifier", "KMeans", "PCA"],
       correctAnswer: 0,
       explanation: "GridSearchCV is a function in Scikit-learn that can be used to tune the hyperparameters of a machine learning model. RandomForestClassifier, KMeans, and PCA are specific machine learning models and techniques."
     },
     {
       question: "Which Scikit-learn function can be used to perform text feature extraction?",
       options: ["MinMaxScaler", "LabelEncoder", "StandardScaler", "CountVectorizer"],
       correctAnswer: 3,
       explanation: "CountVectorizer is a function in Scikit-learn that can be used to perform text feature extraction. LabelEncoder, MinMaxScaler, and StandardScaler are used for encoding categorical features and feature scaling."
     },
     {
       question: "Which Scikit-learn function can be used to visualize the decision boundaries of a classification model?",
       options: ["plot_decision_regions", "plot_confusion_matrix", "plot_roc_curve", "plot_learning_curve"],
       correctAnswer: 0,
       explanation: "The plot_decision_regions function in Scikit-learn can be used to visualize the decision boundaries of a classification model. plot_confusion_matrix, plot_roc_curve, and plot_learning_curve are used for other types of visualizations."
     },
     {
       question: "Which Scikit-learn function can be used to perform K-Means clustering?",
       options: ["KMeans", "RandomForestClassifier", "PCA", "LinearRegression"],
       correctAnswer: 0,
       explanation: "The KMeans function in Scikit-learn can be used to perform K-Means clustering. RandomForestClassifier, PCA, and LinearRegression are other machine learning models."
     },
     {
       question: "Which Scikit-learn function can be used to perform feature selection?",
       options: ["LogisticRegression", "GridSearchCV", "RandomForestClassifier", "SelectKBest"],
       correctAnswer: 3,
       explanation: "CountVectorizer is a function in Scikit-learn that can be used to perform text feature extraction. LabelEncoder, MinMaxScaler, and StandardScaler are used for encoding categorical features and feature scaling."
     },
     {
       question: "Which Scikit-learn function can be used to perform feature selection?",
       options: ["LogisticRegression", "GridSearchCV", "RandomForestClassifier", "SelectKBest"],
       correctAnswer: 3,
       explanation: "CountVectorizer is a function in Scikit-learn that can be used to perform text feature extraction. LabelEncoder, MinMaxScaler, and StandardScaler are used for encoding categorical features and feature scaling."
     },
     {
       question: "Which Scikit-learn function can be used to perform feature selection?",
       options: ["LogisticRegression", "GridSearchCV", "RandomForestClassifier", "SelectKBest"],
       correctAnswer: 3,
       explanation: "CountVectorizer is a function in Scikit-learn that can be used to perform text feature extraction. LabelEncoder, MinMaxScaler, and StandardScaler are used for encoding categorical features and feature scaling."
     },
     {
       question: "Which Scikit-learn function can be used to perform feature selection?",
       options: ["LogisticRegression", "GridSearchCV", "RandomForestClassifier", "SelectKBest"],
       correctAnswer: 3,
       explanation: "CountVectorizer is a function in Scikit-learn that can be used to perform text feature extraction. LabelEncoder, MinMaxScaler, and StandardScaler are used for encoding categorical features and feature scaling."
     },
];

const questionEl = document.getElementById("question");
const option1El = document.getElementById("option1");
const option2El = document.getElementById("option2");
const option3El = document.getElementById("option3");
const option4El = document.getElementById("option4");
const checkBtn = document.getElementById("check");
const nextBtn = document.getElementById("next");
const ansDiv = document.getElementById("show-ans");

let currentQuestion = 0;
let score = 0;

displayQuestion(currentQuestion);

checkBtn.addEventListener("click", function() {
     checkAnswer();
     checkBtn.style.display = "none";
     nextBtn.style.display = "block";
});

function displayQuestion(questionIndex) {
     const question = questions[questionIndex];
     questionEl.innerHTML = question.question;
     option1El.innerHTML = question.options[0];
     option2El.innerHTML = question.options[1];
     option3El.innerHTML = question.options[2];
     option4El.innerHTML = question.options[3];
}

function checkAnswer() {
     const selectedOption = document.querySelector("input[type=radio]:checked");
     if (!selectedOption) {
          ansDiv.innerHTML = "Please select an option.";
          ansDiv.style.display = "block";
          ansDiv.style.borderColor = "black";
          return;
     }
   
     const answer = selectedOption.id;
     const actualAnswer = "ans" + (questions[currentQuestion].correctAnswer+1);
     if (answer == actualAnswer) {
          ansDiv.innerHTML = "Correct! " + questions[currentQuestion].explanation;
          ansDiv.style.display = "block";
          ansDiv.style.borderColor = "rgb(0, 200, 0)";
          score++;
     } else {
          ansDiv.innerHTML = "Incorrect. " + questions[currentQuestion].explanation;
          ansDiv.style.display = "block";
          ansDiv.style.borderColor = "crimson";
     }
}

nextBtn.addEventListener("click", () => {
     currentQuestion++;
     if (currentQuestion < questions.length) {
          displayQuestion(currentQuestion);
          deselectAnswers();
          ansDiv.style.display = "none";
          checkBtn.style.display = "block";
          nextBtn.style.display = "none";
     } else {
          displayScore();
     }
});

function displayScore() {
     const container = document.getElementById("container");
     container.innerHTML = "<h2> You scored <br>" + score + " out of " + questions.length + "! </h2>";
}

function deselectAnswers() {
     document.getElementById("ans1").checked = false;
     document.getElementById("ans2").checked = false;
     document.getElementById("ans3").checked = false;
     document.getElementById("ans4").checked = false;
}
   