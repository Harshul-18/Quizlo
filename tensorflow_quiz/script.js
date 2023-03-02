const questions = [
  {
    question: "What is tensorflow and what are its primary features?",
    code: "tensorflow",
    options: [
      "Tensorflow is a deep learning library that allows for efficient computation on large datasets.",
      "Tensorflow is a machine learning library that allows for efficient computation on small datasets.",
      "Tensorflow is a natural language processing library that allows for efficient computation on large datasets",
      "Tensorflow is a natural language processing library that allows for efficient computation on small datasets",
    ],
    correctAnswer: 0,
    explanation: "Tensorflow is an open sourcelibrary for numerical computation and machine learning. It allows for efficient computation on large datasets, and is widely used for deep learning tasks."
  },
  {
    question: "What is tensorflow and what are its primary features?",
    code: "",
    options: [
      "Tensorflow is a deep learning library that allows for efficient computation on large datasets.",
      "Tensorflow is a machine learning library that allows for efficient computation on small datasets.",
      "Tensorflow is a natural language processing library that allows for efficient computation on large datasets",
      "Tensorflow is a natural language processing library that allows for efficient computation on small datasets",
    ],
    correctAnswer: 0,
    explanation: "Tensorflow is an open sourcelibrary for numerical computation and machine learning. It allows for efficient computation on large datasets, and is widely used for deep learning tasks."
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
   