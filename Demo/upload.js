const file = document.getElementById("file");
const button = document.getElementById("btn");
const fileName = document.getElementById("file-name");

const popupSuccess = document.querySelector(".popup-success");
const popupFailed = document.querySelector(".popup-failed");

const popupTextSuccess = document.getElementById("popup-text-success");
const popupTextFailed = document.getElementById("popup-text-failed");

const closeBtn = document.getElementById("close-btn");
const nextBtn = document.getElementById("next-btn");

const loader = document.getElementById("loader");

file.addEventListener("change", function(){
    if (file.files.length > 0) {
        fileName.textContent = file.files[0].name;
    } else {
        fileName.textContent = "No file selected"
    }
})
button.addEventListener("click", async function() {
    if (file.files.length == 0) {
        popupTextFailed.textContent = "Please select a PDF file.";
        popupFailed.style.display = "block";  
        return;
    }

    loader.style.display = "flex";

    try {
        const formData = new FormData();
        formData.append("file", file.files[0]);
        const response = await fetch("http://127.0.0.1:8000/upload", {
            method: "POST",
            body: formData
        });
        if (response.ok) {
            const data = await response.json();
            popupTextSuccess.textContent = data.message;
            popupSuccess.style.display = "block";
        } else {
            popupTextFailed.textContent = "Error: Unsuccessful Indexing.";
            popupFailed.style.display = "block";
        }
    } catch (error) {
        console.error("Error uploading file:", error);
        popupTextFailed.textContent = "Error: Failed to upload file.";
        popupFailed.style.display = "block";
    } finally {
        loader.style.display = "none";
    }
})
closeBtn.addEventListener("click", function() {
    popupFailed.style.display = "none";
})
nextBtn.addEventListener("click", function() {
    popupSuccess.style.display = "none";
    window.location.href = "chat page.html";
})
