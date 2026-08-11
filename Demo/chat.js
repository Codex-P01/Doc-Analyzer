const backButton = document.getElementById("back-btn");
const sendBtn = document.getElementById("send-btn");
const messageInput = document.getElementById("message");
const chatMessages = document.getElementById("chat-messages");

const summaryBtn = document.getElementById("summary-btn");

const loader = document.getElementById("loader");
const headText = document.getElementById("head-text");
const paraText = document.getElementById("para-text");

const popupText = document.getElementById("popup-text-summary");
const popup = document.querySelector(".popup-summary");
const clsBtn = document.getElementById("close-btn");

sendBtn.addEventListener("click", async () => {
    messageIP = messageInput.value.trim();
    if (messageIP === "") {
        return;
    }
    headText.textContent = "Loading...";
    paraText.textContent = "Please wait while we process your question";
    loader.style.display = "flex";
    try {
        const response = await fetch("http://127.0.0.1:8000/query", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                query: messageIP
            })
        });
        const data = await response.json();
        textMessage = data.answer;
        chatMessages.innerHTML += `<div class="user-message">
                                    <p>${messageIP}</p>
                                </div>`;
        chatMessages.innerHTML += `<div class="bot-message">
                                    <p>${textMessage}</p>
                                </div>`;
        messageInput.value = "";
    } catch (error) {
        console.error("Error fetching chat response:", error);
    } finally {
        loader.style.display = "none";
    }
});

summaryBtn.addEventListener("click", async () => {
    headText.textContent = "Processing...";
    paraText.textContent = "Please wait while we process your request";
    loader.style.display = "flex";
    try {
        const response = await fetch("http://127.0.0.1:8000/summary", {
            method: "GET",
            headers: {
                "Content-Type": "application/json"
            }
        });
        const data = await response.json();
        popupText.textContent = data.summary;
        popup.style.display = "block";
    } catch (error) {
        console.error("Error fetching summary:", error);
    } finally {
        loader.style.display = "none";
    }
});

backButton.addEventListener("click", function () {
    window.location.href = "upload page.html";
});

clsBtn.addEventListener("click", () =>{
    popup.style.display = "none";
})