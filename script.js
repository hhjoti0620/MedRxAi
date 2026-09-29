```javascript
// ============================================================
// MedRxAI - Frontend JavaScript
// Model 1 + Model 2 Image Upload & Inference
// ============================================================

document.addEventListener("DOMContentLoaded", () => {

    // Initialize both model upload handlers
    initializeUpload(1);
    initializeUpload(2);

});


// ============================================================
// INITIALIZE UPLOAD
// ============================================================

function initializeUpload(modelNumber) {

    const input = document.getElementById(
        `image-input-${modelNumber}`
    );

    const uploadArea = document.getElementById(
        `upload-area-${modelNumber}`
    );

    const fileInfo = document.getElementById(
        `file-info-${modelNumber}`
    );

    const fileName = document.getElementById(
        `file-name-${modelNumber}`
    );

    if (!input || !uploadArea) {
        console.error(
            `Upload elements for Model ${modelNumber} not found.`
        );
        return;
    }


    // --------------------------------------------------------
    // File selected
    // --------------------------------------------------------

    input.addEventListener("change", function () {

        if (this.files && this.files.length > 0) {

            const file = this.files[0];

            if (!validateImage(file)) {
                this.value = "";
                return;
            }

            fileName.textContent = file.name;

            fileInfo.style.display = "flex";

            uploadArea.classList.add("has-file");

        }

    });


    // --------------------------------------------------------
    // Drag & Drop
    // --------------------------------------------------------

    uploadArea.addEventListener("dragover", (event) => {

        event.preventDefault();

        uploadArea.classList.add("drag-over");

    });


    uploadArea.addEventListener("dragleave", () => {

        uploadArea.classList.remove("drag-over");

    });


    uploadArea.addEventListener("drop", (event) => {

        event.preventDefault();

        uploadArea.classList.remove("drag-over");

        const files = event.dataTransfer.files;

        if (!files || files.length === 0) {
            return;
        }

        const file = files[0];

        if (!validateImage(file)) {
            return;
        }

        // Assign dropped file to input
        const dataTransfer = new DataTransfer();

        dataTransfer.items.add(file);

        input.files = dataTransfer.files;

        fileName.textContent = file.name;

        fileInfo.style.display = "flex";

        uploadArea.classList.add("has-file");

    });

}



// ============================================================
// IMAGE VALIDATION
// ============================================================

function validateImage(file) {

    const allowedTypes = [
        "image/jpeg",
        "image/jpg",
        "image/png"
    ];

    if (!allowedTypes.includes(file.type)) {

        showNotification(
            "Please upload a JPG, JPEG or PNG image.",
            "error"
        );

        return false;
    }


    // Maximum file size = 10 MB

    const maxSize = 10 * 1024 * 1024;

    if (file.size > maxSize) {

        showNotification(
            "Image size must be less than 10 MB.",
            "error"
        );

        return false;
    }

    return true;

}



// ============================================================
// REMOVE SELECTED FILE
// ============================================================

function removeFile(modelNumber) {

    const input = document.getElementById(
        `image-input-${modelNumber}`
    );

    const fileInfo = document.getElementById(
        `file-info-${modelNumber}`
    );

    const uploadArea = document.getElementById(
        `upload-area-${modelNumber}`
    );

    const result = document.getElementById(
        `result-${modelNumber}`
    );


    if (input) {
        input.value = "";
    }

    if (fileInfo) {
        fileInfo.style.display = "none";
    }

    if (uploadArea) {
        uploadArea.classList.remove("has-file");
    }

    if (result) {
        result.style.display = "none";
    }

}



// ============================================================
// RUN MODEL
// ============================================================

async function runModel(modelNumber) {

    const input = document.getElementById(
        `image-input-${modelNumber}`
    );

    const runButton = document.getElementById(
        `run-btn-${modelNumber}`
    );

    const loading = document.getElementById(
        `loading-${modelNumber}`
    );

    const result = document.getElementById(
        `result-${modelNumber}`
    );

    const resultImage = document.getElementById(
        `result-image-${modelNumber}`
    );


    // --------------------------------------------------------
    // Check image
    // --------------------------------------------------------

    if (!input || !input.files || input.files.length === 0) {

        showNotification(
            `Please upload a prescription image for Model ${modelNumber}.`,
            "error"
        );

        return;
    }


    const file = input.files[0];


    if (!validateImage(file)) {
        return;
    }


    // --------------------------------------------------------
    // Create FormData
    // --------------------------------------------------------

    const formData = new FormData();

    formData.append("image", file);


    // --------------------------------------------------------
    // Select Flask endpoint
    // --------------------------------------------------------

    let endpoint = "";

    if (modelNumber === 1) {

        endpoint = "/predict/model1";

    } else if (modelNumber === 2) {

        endpoint = "/predict/model2";

    } else {

        showNotification(
            "Invalid model selected.",
            "error"
        );

        return;
    }


    // --------------------------------------------------------
    // UI: Loading
    // --------------------------------------------------------

    if (runButton) {

        runButton.disabled = true;

        runButton.dataset.originalText =
            runButton.textContent;

        runButton.textContent =
            "Processing...";

    }


    if (loading) {
        loading.style.display = "flex";
    }


    if (result) {
        result.style.display = "none";
    }


    try {

        // ----------------------------------------------------
        // Send image to Flask
        // ----------------------------------------------------

        const response = await fetch(
            endpoint,
            {
                method: "POST",
                body: formData
            }
        );


        // ----------------------------------------------------
        // Read response
        // ----------------------------------------------------

        const contentType =
            response.headers.get("content-type") || "";


        let data;


        if (contentType.includes("application/json")) {

            data = await response.json();

        } else {

            const text = await response.text();

            throw new Error(
                text || "Unexpected server response."
            );

        }


        // ----------------------------------------------------
        // Backend error
        // ----------------------------------------------------

        if (!response.ok || data.success === false) {

            throw new Error(
                data.error ||
                data.message ||
                "Model inference failed."
            );

        }


        // ----------------------------------------------------
        // Display result image
        // ----------------------------------------------------

        if (data.result_image) {

            resultImage.src =
                data.result_image +
                "?t=" +
                new Date().getTime();

            result.style.display = "block";

        }


        // ----------------------------------------------------
        // Display detection information
        // ----------------------------------------------------

        displayDetectionResults(
            modelNumber,
            data
        );


        showNotification(
            `Model ${modelNumber} completed successfully.`,
            "success"
        );


    } catch (error) {

        console.error(
            `Model ${modelNumber} error:`,
            error
        );


        showNotification(
            error.message ||
            "Something went wrong while processing the image.",
            "error"
        );

    } finally {

        // ----------------------------------------------------
        // Restore UI
        // ----------------------------------------------------

        if (runButton) {

            runButton.disabled = false;

            runButton.textContent =
                runButton.dataset.originalText ||
                `Run Model ${modelNumber}`;

        }


        if (loading) {
            loading.style.display = "none";
        }

    }

}



// ============================================================
// DISPLAY DETECTION RESULTS
// ============================================================

function displayDetectionResults(modelNumber, data) {

    const resultContainer =
        document.getElementById(
            `result-${modelNumber}`
        );


    if (!resultContainer) {
        return;
    }


    // Remove previous result information

    const oldInfo =
        resultContainer.querySelector(
            ".detection-info"
        );

    if (oldInfo) {
        oldInfo.remove();
    }


    // If backend sends detections

    if (
        !data.detections ||
        !Array.isArray(data.detections) ||
        data.detections.length === 0
    ) {

        return;
    }


    const infoDiv =
        document.createElement("div");

    infoDiv.className =
        "detection-info";


    const title =
        document.createElement("h5");

    title.textContent =
        `Detected Objects (${data.detections.length})`;


    infoDiv.appendChild(title);


    const list =
        document.createElement("div");

    list.className =
        "detection-list";


    data.detections.forEach(
        (detection, index) => {

            const item =
                document.createElement("div");

            item.className =
                "detection-item";


            const name =
                detection.name ||
                detection.class ||
                detection.label ||
                `Object ${index + 1}`;


            const confidence =
                detection.confidence !== undefined
                    ? `${(
                        Number(detection.confidence) * 100
                    ).toFixed(1)}%`
                    : "";


            item.innerHTML = `
                <span class="detection-name">
                    ${escapeHTML(String(name))}
                </span>

                ${
                    confidence
                    ? `<span class="confidence">
                         ${confidence}
                       </span>`
                    : ""
                }
            `;


            list.appendChild(item);

        }
    );


    infoDiv.appendChild(list);

    resultContainer.appendChild(infoDiv);

}



// ============================================================
// NOTIFICATION
// ============================================================

function showNotification(message, type = "info") {

    // Remove previous notification

    const existing =
        document.querySelector(
            ".medrx-notification"
        );

    if (existing) {
        existing.remove();
    }


    // Create notification

    const notification =
        document.createElement("div");

    notification.className =
        `medrx-notification ${type}`;


    notification.innerHTML = `
        <span class="notification-message">
            ${escapeHTML(String(message))}
        </span>

        <button
            type="button"
            class="notification-close"
            aria-label="Close">
            ×
        </button>
    `;


    document.body.appendChild(
        notification
    );


    // Close button

    const closeButton =
        notification.querySelector(
            ".notification-close"
        );


    closeButton.addEventListener(
        "click",
        () => {
            notification.remove();
        }
    );


    // Automatically remove

    setTimeout(() => {

        if (
            document.body.contains(notification)
        ) {

            notification.remove();

        }

    }, 5000);

}



// ============================================================
// HTML ESCAPE
// ============================================================

function escapeHTML(value) {

    const div =
        document.createElement("div");

    div.textContent = value;

    return div.innerHTML;

}



// ============================================================
// SMOOTH SCROLL
// ============================================================

document.addEventListener(
    "click",
    function (event) {

        const link =
            event.target.closest(
                'a[href^="#"]'
            );

        if (!link) {
            return;
        }


        const targetId =
            link.getAttribute("href");


        if (
            !targetId ||
            targetId === "#"
        ) {
            return;
        }


        const target =
            document.querySelector(targetId);


        if (target) {

            event.preventDefault();

            target.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        }

    }
);



// ============================================================
// PREVENT FORM DEFAULT SUBMISSION
// ============================================================

document.addEventListener(
    "submit",
    function (event) {

        event.preventDefault();

    }
);
```
