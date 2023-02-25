const questions = [
     {
       question: "What is the capital of India?",
       options: ["Mumbai", "New Delhi", "Kolkata", "Chennai"],
       correctAnswer: 1,
       explanation: "New Delhi is the capital of India."
     },
     {
       question: "What is the currency of Japan?",
       options: ["Yuan", "Yen", "Dollar", "Euro"],
       correctAnswer: 1,
       explanation: "Yen is the currency of Japan."
     },
     {
       question: "What is the national animal of Australia?",
       options: ["Kangaroo", "Tasmanian devil", "Koala", "Emu"],
       correctAnswer: 0,
       explanation: "Kangaroo is the national animal of Australia."
     }
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
   