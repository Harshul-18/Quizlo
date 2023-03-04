const questions = [
     {
     question: "What is the purpose of the activation function in a neural network?",
     code: "",
     options: [
     "To increase the size of the input data",
     "To reduce the size of the output data",
     "To add non-linearity to the neural network",
     "To normalize the input data"
     ],
     correctAnswer: 2,
     explanation: "Activation functions are used in neural networks to introduce non-linearity to the model. Without non-linearity, a neural network would just be a linear regression model, which is not capable of learning complex patterns and relationships in the data."
     },
     {
     question: "What is the difference between a convolutional layer and a fully connected layer in a neural network?",
     code: "",
     options: [
     "Convolutional layers are used for images, while fully connected layers are used for text",
     "Convolutional layers are used for spatial data, while fully connected layers are used for non-spatial data",
     "Convolutional layers are used for classification, while fully connected layers are used for regression",
     "There is no difference between convolutional and fully connected layers"
     ],
     correctAnswer: 1,
     explanation: "Convolutional layers are typically used for image and other spatial data, where there is a strong spatial relationship between the input features. Fully connected layers, on the other hand, are used for non-spatial data where there is no inherent spatial relationship between the input features."
     },
     {
     question: "What is the purpose of dropout regularization in a neural network?",
     code: "",
     options: [
     "To increase the size of the input data",
     "To reduce the size of the output data",
     "To add non-linearity to the neural network",
     "To prevent overfitting"
     ],
     correctAnswer: 3,
     explanation: "Dropout regularization is a technique used to prevent overfitting in neural networks. It works by randomly dropping out (setting to zero) a percentage of the neurons in a layer during training, which forces the remaining neurons to learn more robust features that are not dependent on the presence of any particular neuron."
     },
     {
     question: "What is the purpose of the loss function in a neural network?",
     code: "",
     options: [
     "To measure the performance of the neural network",
     "To adjust the weights of the neural network",
     "To preprocess the input data",
     "To generate new data for the neural network"
     ],
     correctAnswer: 0,
     explanation: "The loss function is used to measure the performance of the neural network by computing the difference between the predicted output and the actual output. The goal of training a neural network is to minimize the loss function, which is accomplished by adjusting the weights of the network."
     },
     {
     question: "What is the purpose of the optimizer in a neural network?",
     code: "",
     options: [
     "To measure the performance of the neural network",
     "To adjust the weights of the neural network",
     "To preprocess the input data",
     "To generate new data for the neural network"
     ],
     correctAnswer: 1,
     explanation: "The optimizer is used to adjust the weights of the neural network during training in order to minimize the loss function. There are many different types of optimizers available in TensorFlow/Keras, each with its own strengths and weaknesses."
     },
     {
     question: "What is the purpose of early stopping in a neural network?",
     code: "",
     options: [
     "To prevent overfitting",
     "To speed up training",
     "To increase the size of the input data",
     "To reduce the size of the output data"
     ],
     correctAnswer: 0,
     explanation: "Early stopping is a technique used to prevent overfitting in a neural network. It works by monitoring the validation loss during training and stopping the training process when the validation loss starts to increase. This helps prevent the model from memorizing the training data and not generalizing well to new data."
     },
     {
     question: "What is transfer learning in deep learning?",
     code: "",
     options: [
     "A technique for transferring weights from one neural network to another",
     "A technique for transferring data between different domains",
     "A technique for transferring features learned from one task to another",
     "A technique for transferring input data between different models"
     ],
     correctAnswer: 2,
     explanation: "Transfer learning is a technique in deep learning where a model trained on one task is reused as a starting point for training a model on a different task. This is often done by freezing the weights of the early layers of the pre-trained model and fine-tuning the later layers for the new task."
     },
     {
     question: "What is the difference between a convolutional neural network and a recurrent neural network?",
     code: "",
     options: [
     "Convolutional neural networks are used for images, while recurrent neural networks are used for text",
     "Convolutional neural networks are used for spatial data, while recurrent neural networks are used for temporal data",
     "Convolutional neural networks are used for classification, while recurrent neural networks are used for regression",
     "There is no difference between convolutional and recurrent neural networks"
     ],
     correctAnswer: 1,
     explanation: "Convolutional neural networks are typically used for image and other spatial data, while recurrent neural networks are used for sequential or temporal data, such as text or time series data. RNNs are able to remember previous inputs and use that information to inform the next prediction, which makes them well-suited for tasks such as language modeling or speech recognition."
     },
     {
     question: "What is batch normalization in a neural network?",
     code: "",
     options: [
     "A technique for normalizing the input data",
     "A technique for normalizing the output data",
     "A technique for normalizing the weights of the neural network",
     "A technique for normalizing the activations of the neural network"
     ],
     correctAnswer: 3,
     explanation: "Batch normalization is a technique used to normalize the activations of a neural network by normalizing the output of each layer. This helps prevent the activations from becoming too large or too small, which can make training difficult. Batch normalization is often used in conjunction with other regularization techniques, such as dropout, to improve the performance of the network."
     },
     {
     question: "What is the output of the following code?",
     code: `import tensorflow as tf <br> <br> x = tf.constant([1, 2, 3, 4]) <br> y = tf.constant([5, 6, 7, 8]) <br> <br> z = tf.concat([x, y], axis=0) <br> <br> print(z.shape) <br> `,
     options: [
     "(8,)",
     "(4,)",
     "(2, 4)",
     "None of the above",
     ],
     correctAnswer: 0,
     explanation: "The output of the code is a one-dimensional tensor with shape (8,). The concat() function is used to concatenate two tensors along a given axis. In this case, the axis is 0, so the two tensors are stacked vertically."
     },
     {
     question: "What is the output of the following code?",
     code: `import tensorflow as tf <br> <br> x = tf.constant([[1, 2], [3, 4]]) <br> y = tf.constant([[5, 6], [7, 8]]) <br> <br> z = tf.matmul(x, y) <br> <br> print(z.shape) <br> `,
     options: [
     "(2, 2)",
     "(2, 1)",
     "(1, 2)",
     "(2, 3)"
     ],
     correctAnswer: 0,
     explanation: "The output of the code is a two-dimensional tensor with shape (2, 2). The matmul() function is used to perform matrix multiplication between two tensors. In this case, the two matrices are multiplied and the result is a new matrix with shape (2, 2)."
     },
     {
     question: "What is the output of the following code?",
     code: `import tensorflow as tf <br> <br> x = tf.constant([1, 2, 3, 4]) <br> <br> y = tf.square(x) <br> <br> print(y.numpy()) <br> `,
     options: [
     "[1, 4, 9, 16]",
     "[1, 8, 27, 64]",
     "[2, 4, 6, 8]",
     "[1, 2, 3, 4]"
     ],
     correctAnswer: 0,
     explanation: "The output of the code is an array of squared values of x, which is [1, 4, 9, 16]. The square() function is used to calculate the square of each element in the tensor x. The numpy() function is used to convert the resulting tensor to a NumPy array."
     },
     {
     question: "What is the output of the following code?",
     code: `import tensorflow as tf <br> <br> x = tf.constant([1, 2, 3, 4]) <br> <br> y = tf.reduce_sum(x) <br> <br> print(y.numpy()) <br> `,
     options: [
     "10",
     "24",
     "6",
     "4"
     ],
     correctAnswer: 0,
     explanation: "The output of the code is the sum of all the elements in the tensor x, which is 10. The reduce_sum() function is used to calculate the sum of all the elements in the tensor."
     },
     {
     question: "What is the output of the following code?",
     code: `import tensorflow as tf <br> <br> x = tf.constant([1, 2, 3, 4]) <br> <br> y = tf.reshape(x, (2, 2)) <br> <br> print(y.numpy()) <br> `,
     options: [
     "[[1, 2], [3, 4]]",
     "[1, 2, 3, 4]",
     "[[1], [2], [3], [4]]",
     "[[1, 2, 3], [4]]"
     ],
     correctAnswer: 0,
     explanation: "The output of the code is a two-dimensional tensor with shape (2, 2), where the elements are reshaped from the one-dimensional tensor x. The reshape() function is used to reshape the tensor x into a new shape (2, 2)."
     },
     {
     question: "What is the output of the following code?",
     code: `import tensorflow as tf <br> <br> x = tf.constant([[1, 2], [3, 4]]) <br> y = tf.constant([10, 20]) <br> <br> z = tf.add(x, y) <br> <br> print(z.numpy()) <br> `,
     options: [
     "[[11, 22], [13, 24]]",
     "[[1, 2], [3, 4], [10, 20]]",
     "[[11], [22], [13], [24]]",
     "None of the above"
     ],
     correctAnswer: 0,
     explanation: "The output of the code is a two-dimensional tensor with shape (2, 2), where each element of x is added to the corresponding element of y. The add() function is used to add two tensors element-wise."
     },
     {
     question: "What is the output of the following code?",
     code: `import tensorflow as tf <br> <br> x = tf.constant([[1, 2], [3, 4]]) <br> y = tf.constant([[10], [20]]) <br> <br> z = tf.matmul(x, y) <br> <br> print(z.numpy()) <br> `,
     options: [
     "[[50], [110]]",
     "[[10, 20], [30, 40]]",
     "[[10], [20], [30], [40]]",
     "[[11, 22], [13, 24]]"
     ],
     correctAnswer: 0,
     explanation: "The output of the code is a two-dimensional tensor with shape (2, 1), which is the result of matrix multiplication between x and y. The matmul() function is used to perform matrix multiplication between two tensors."
     },
     {
     question: "What is the purpose of activation functions in deep learning?",
     code: "",
     options: [
     "To adjust weights in the neural network",
     "To prevent overfitting",
     "To introduce non-linearity into the model",
     "To compute the loss function",
     ],
     correctAnswer: 2,
     explanation: "Activation functions introduce non-linearity into the model by applying a mathematical function to the output of a neuron. This enables the network to learn complex relationships between inputs and outputs."
     },
     {
     question: "What is the difference between a dense layer and a convolutional layer in Keras?",
     code: "",
     options: [
     "A dense layer is used for classification tasks while a convolutional layer is used for regression tasks",
     "A dense layer is fully connected while a convolutional layer is partially connected",
     "A dense layer is more efficient than a convolutional layer",
     "There is no difference between a dense layer and a convolutional layer",
     ],
     correctAnswer: 1,
     explanation: "A dense layer is fully connected, meaning that each neuron in the layer is connected to every neuron in the previous layer. In contrast, a convolutional layer is partially connected, meaning that each neuron is only connected to a small subset of neurons in the previous layer. Convolutional layers are often used in image processing tasks while dense layers are used for classification tasks."
     },
     {
     question: "What is the purpose of dropout regularization in deep learning?",
     code: "",
     options: [
     "To speed up training",
     "To prevent overfitting",
     "To reduce the size of the model",
     "To increase the accuracy of the model",
     ],
     correctAnswer: 1,
     explanation: "Dropout regularization is a technique used to prevent overfitting by randomly dropping out neurons during training. This helps the network to learn more robust features by preventing neurons from relying too heavily on each other."
     },
     {
     question: "What is the difference between binary and categorical cross-entropy loss in Keras?",
     code: "",
     options: [
     "There is no difference between binary and categorical cross-entropy loss",
     "Binary cross-entropy is used for binary classification while categorical cross-entropy is used for multi-class classification",
     "Binary cross-entropy is used for regression while categorical cross-entropy is used for classification",
     "Categorical cross-entropy is used for binary classification while binary cross-entropy is used for multi-class classification",
     ],
     correctAnswer: 1,
     explanation: "Binary cross-entropy is used for binary classification tasks while categorical cross-entropy is used for multi-class classification tasks. Binary cross-entropy is a special case of categorical cross-entropy where there are only two classes."
     },
     {
     question: "What is the purpose of the optimizer in Keras?",
     code: "",
     options: [
     "To compute the loss function",
     "To adjust the weights in the neural network",
     "To preprocess the data",
     "To monitor the training process",
     ],
     correctAnswer: 1,
     explanation: "The optimizer is responsible for adjusting the weights in the neural network during training in order to minimize the loss function. There are many different types of optimizers available in Keras, each with their own strengths and weaknesses."
     },
     {
     question: "What is the purpose of the validation set in Keras?",
     code: null,
     options: [
     "To train the model",
     "To evaluate the performance of the model on unseen data",
     "To prevent overfitting",
     "To provide a test set for the model",
     ],
     correctAnswer: 2,
     explanation: "The validation set is used to evaluate the performance of the model on unseen data during training. It is used to monitor the generalization of the model and to prevent overfitting. The test set is typically reserved for the final evaluation of the model after it has been trained."
     },
     {
     question: "What is the purpose of the learning rate in Keras?",
     code: null,
     options: [
     "To adjust the weights in the neural network",
     "To prevent overfitting",
     "To control the speed of learning during training",
     "To compute the loss function",
     ],
     correctAnswer: 2,
     explanation: "The learning rate controls the speed of learning during training. A larger learning rate will result in faster convergence but may cause the model to overshoot the optimal solution, while a smaller learning rate will result in slower convergence but may help the model to avoid local minima."
     },
     {
     question: "What is the difference between L1 and L2 regularization in Keras?",
     code: null,
     options: [
     "L1 regularization penalizes the absolute value of the weights while L2 regularization penalizes the square of the weights",
     "L1 regularization is used for classification tasks while L2 regularization is used for regression tasks",
     "L1 regularization is more efficient than L2 regularization",
     "There is no difference between L1 and L2 regularization",
     ],
     correctAnswer: 0,
     explanation: "L1 regularization penalizes the absolute value of the weights in the neural network, while L2 regularization penalizes the square of the weights. L1 regularization can be used to enforce sparsity in the model by encouraging some weights to be exactly zero."
     },
     {
     question: "What is the purpose of a callback in Keras?",
     code: null,
     options: [
     "To preprocess the data",
     "To monitor the training process",
     "To adjust the weights in the neural network",
     "To compute the loss function",
     ],
     correctAnswer: 1,
     explanation: "A callback is a function that can be called during training to monitor the training process or to modify the behavior of the model. Examples of callbacks include EarlyStopping, which can be used to stop training early if the validation loss stops improving, and ModelCheckpoint, which can be used to save the weights of the model at certain points during training."
     },
     {
     question: "What is the purpose of transfer learning in deep learning?",
     code: null,
     options: [
     "To transfer weights from one model to another",
     "To prevent overfitting",
     "To preprocess the data",
     "To reuse pre-trained models to solve related tasks",
     ],
     correctAnswer: 3,
     explanation: "Transfer learning is a technique in deep learning where a pre-trained model is used as a starting point for a new model that is designed to solve a related task. By reusing the weights learned by the pre-trained model, the new model can be trained more efficiently and with less data than would otherwise be required."
     },
     {
     question: "What is the purpose of the following code snippet?",
     code: `model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) <br>`,
     options: [
     "To compile the model with a categorical cross-entropy loss, accuracy metric and an Adam optimizer",
     "To compile the model with a binary cross-entropy loss, accuracy metric and an Adam optimizer",
     "To compile the model with a mean squared error loss, accuracy metric and an Adam optimizer",
     "To compile the model with a mean absolute error loss, accuracy metric and an Adam optimizer",
     ],
     correctAnswer: 0,
     explanation: "The <code>compile</code> function in Keras is used to configure the learning process of a model. In the code snippet above, the <code>loss</code> argument specifies the loss function to use, the <code>optimizer</code> argument specifies the optimizer to use, and the <code>metrics</code> argument specifies the metrics to use. The <code>categorical_crossentropy</code> loss function is commonly used for multiclass classification tasks, while the <code>binary_crossentropy</code> loss function is used for two class classification tasks. The <code>adam</code> optimizer is popular optimization algorithm for training neural networks. The metric used is <code>accuracy</code>."
     },
     {
     question: "What is the difference between a dense layer and a convolutional layer in Keras?",
     options: [
     "A dense layers is used for convolutional neural netoworks, while a convolutional layer is used for fully connected neural networks.",
     "A dense layer performs element-wise multiplication between the inputs and weights, while a convolutional layer performs a dot product between the inputs and a set of learnable filters.",
     "A dense layer applies a fixed-size filter to the input data, while a convolutional layer applies filters of varying sizes.",
     "A dense layer is used for feature extraction, while a convolutional layer is used for classification.",
     ],
     correctAnswer: 1,
     explanation: "In Keras, a dense layer is a fully connected layer, where each neuron in the layer is connected to every neuron in the previous layer. A dense layer performs a matrix multiplication between the input data and a set of learnable weights, followed by an element-wise activation function. In contrast, a convolutional layer applies a set of learnable filters to a small region of the input data at a time, performing a dot product between the filter weights and the input data. Convolutional layers are commonly used for image and signal processing tasks, where local patterns in the input data are important.",
     },
     {
     question: "What is the output type of the following code?",
     code: `
     import tensorflow as tf <br>
     <br>
     x = tf.constant(3) <br>
     y = tf.constant(4) <br>
     <br>
     z = x + y <br>
     <br>
     print(type(z)) <br>
     `,
     options: [
          "int32",
          "float32",
          "Tensor",
          "None of the above",
     ],
     correctAnswer: 2,
     explanation: "The output type of the code is a TensorFlow tensor object."
     },
];

const questionEl = document.getElementById("question");
const questionCodeEl = document.getElementById("question-code");
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
     if (question.code.length > 0) {
      questionCodeEl.style.display = "block";
      questionCodeEl.innerHTML = question.code;
    } else {
      questionCodeEl.style.display = "none";
    }
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