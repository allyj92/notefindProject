// Get the canvas and context
var canvas = document.getElementById("drawingCanvas");
var ctx = canvas.getContext("2d");
var drawing = false;

// Start drawing when the mouse is pressed
canvas.addEventListener("mousedown", function (event) {
  drawing = true;
  ctx.beginPath(); // Start a new path for a fresh stroke
  ctx.moveTo(event.offsetX, event.offsetY); // Move to the current mouse position
});

// Stop drawing when the mouse is released
canvas.addEventListener("mouseup", function () {
  drawing = false;
});

// Draw as the mouse is moved
canvas.addEventListener("mousemove", function (event) {
  if (drawing) {
    ctx.lineWidth = 2; // Set the pen size
    ctx.lineCap = "round"; // Make lines rounded at the ends
    ctx.strokeStyle = "black"; // Set pen color to black
    ctx.lineTo(event.offsetX, event.offsetY); // Draw line to current mouse position
    ctx.stroke(); // Actually draw the line
    ctx.moveTo(event.offsetX, event.offsetY); // Move the starting point to the current position for the next stroke
  }
});

// Function to clear the canvas
function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the entire canvas
}
