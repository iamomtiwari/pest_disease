// frontend/app.js

document.getElementById('predictBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('imageUpload');
    const formData = new FormData();

    if (fileInput.files.length > 0) {
        formData.append('image', fileInput.files[0]);

        // Send a POST request to your backend (implement this in Flask)
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        document.getElementById('result').innerText = `Predicted class: ${result.class}`;
    } else {
        alert("Please upload an image.");
    }
});
