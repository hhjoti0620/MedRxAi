// ============================================================
// MedRxAI - Prescription Upload & Model Inference
// ============================================================

document.addEventListener("DOMContentLoaded", function () {

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
            `Model ${modelNumber} upload elements not found.`
        );
        return;
    }


    // ========================================================
    // FILE SELECT
    // ========================================================

    input.addEventListener("change", function () {

        if (!this.files || this.files.length === 0) {
            return;
        }

        const file = this.files[0];

        if (!validateImage(file)) {
            this.value = "";
            return;
        }

        if (fileName) {
            fileName.textContent = file.name;
        }

        if (fileInfo) {
            fileInfo.style.display = "flex";
        }

        uploadArea.classList.add("has-file");

    });


    // ========================================================
    // DRAG OVER
    // ========================================================

    uploadArea.addEventListener(
        "dragover",
        function (event) {

            event.preventDefault();

            uploadArea.classList.add("drag-over");

        }
    );


    // ========================================================
    // DRAG LEAVE
    // ========================================================

    uploadArea.addEventListener(
        "dragleave",
        function () {

            uploadArea.classList.remove("drag-over");

        }
    );


    // ========================================================
    // DROP
    // ========================================================

    uploadArea.addEventListener(
        "drop",
        function (event) {

            event.preventDefault();

            uploadArea.classList.remove("drag-over");

            const files =
                event.dataTransfer.files;

            if (!files || files.length === 0) {
                return;
            }

            const file = files[0];

            if (!validateImage(file)) {
                return;
            }


            // Put dropped file into input

            const dataTransfer =
                new DataTransfer();

            dataTransfer.items.add(file);

            input.files =
                dataTransfer.files;


            if (fileName) {
                fileName.textContent =
                    file.name;
            }

            if (fileInfo) {
                fileInfo.style.display =
                    "flex";
            }

            uploadArea.classList.add(
                "has-file"
            );

        }
    );

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
            "Please upload JPG, JPEG or PNG image.",
            "error"
        );

        return false;
    }


    // Maximum 10 MB

    const maxSize =
        10 * 1024 * 1024;


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
// REMOVE FILE
// ============================================================

function removeFile(modelNumber) {

    const input =
        document.getElementById(
            `image-input-${modelNumber}`
        );

    const fileInfo =
        document.getElementById(
            `file-info-${modelNumber}`
        );

    const uploadArea =
        document.getElementById(
            `upload-area-${modelNumber}`
        );

    const result =
        document.getElementById(
            `result-${modelNumber}`
        );


    if (input) {
        input.value = "";
    }


    if (fileInfo) {
        fileInfo.style.display =
            "none";
    }


    if (uploadArea) {

        uploadArea.classList.remove(
            "has-file"
        );

    }


    if (result) {

        result.style.display =
            "none";

    }

}



// ============================================================
// RUN MODEL
// ============================================================

async function runModel(modelNumber) {

    const input =
        document.getElementById(
            `image-input-${modelNumber}`
        );

    const runButton =
        document.getElementById(
            `run-btn-${modelNumber}`
        );

    const loading =
        document.getElementById(
            `loading-${modelNumber}`
        );

    const result =
        document.getElementById(
            `result-${modelNumber}`
        );

    const resultImage =
        document.getElementById(
            `result-image-${modelNumber}`
        );


    // ========================================================
    // CHECK IMAGE
    // ========================================================

    if (
        !input ||
        !input.files ||
        input.files.length === 0
    ) {

        showNotification(
            `Please upload a prescription image for Model ${modelNumber}.`,
            "error"
        );

        return;
    }


    const file =
        input.files[0];


    if (!validateImage(file)) {
        return;
    }



    // ========================================================
    // FORM DATA
    // ========================================================

    const formData =
        new FormData();

    formData.append(
        "image",
        file
    );



    // ========================================================
    // MODEL ENDPOINT
    // ========================================================

    let endpoint;


    if (modelNumber === 1) {

        endpoint =
            "/predict/model1";

    }
    else if (modelNumber === 2) {

        endpoint =
            "/predict/model2";

    }
    else {

        showNotification(
            "Invalid model selected.",
            "error"
        );

        return;
    }



    // ========================================================
    // SHOW LOADING
    // ========================================================

    if (runButton) {

        runButton.disabled = true;

        runButton.dataset.originalText =
            runButton.textContent;

        runButton.textContent =
            "Processing...";

    }


    if (loading) {

        loading.style.display =
            "flex";

    }


    if (result) {

        result.style.display =
            "none";

    }



    // ========================================================
    // SEND REQUEST
    // ========================================================

    try {

        const response =
            await fetch(
                endpoint,
                {
                    method: "POST",
                    body: formData
                }
            );


        // ====================================================
        // RESPONSE
        // ====================================================

        const contentType =
            response.headers.get(
                "content-type"
            ) || "";


        let data;


        if (
            contentType.includes(
                "application/json"
            )
        ) {

            data =
                await response.json();

        }
        else {

            const text =
                await response.text();

            throw new Error(
                text ||
                "Unexpected server response."
            );

        }



        // ====================================================
        // ERROR
        // ====================================================

        if (
            !response.ok ||
            data.success === false
        ) {

            throw new Error(
                data.error ||
                data.message ||
                "Model inference failed."
            );

        }



        // ====================================================
        // RESULT IMAGE
        // ====================================================

        if (
            data.result_image &&
            resultImage
        ) {

            resultImage.src =
                data.result_image +
                "?t=" +
                Date.now();


            if (result) {

                result.style.display =
                    "block";

            }

        }



        // ====================================================
        // DETECTION RESULTS
        // ====================================================

        displayDetectionResults(
            modelNumber,
            data
        );


        // ====================================================
        // SUCCESS MESSAGE
        // ====================================================

        showNotification(
            `Model ${modelNumber} completed successfully.`,
            "success"
        );

    }



    // ========================================================
    // ERROR HANDLING
    // ========================================================

    catch (error) {

        console.error(
            `Model ${modelNumber} error:`,
            error
        );


        showNotification(
            error.message ||
            "Something went wrong while processing the image.",
            "error"
        );

    }



    // ========================================================
    // FINALLY
    // ========================================================

    finally {

        if (runButton) {

            runButton.disabled =
                false;

            runButton.textContent =
                runButton.dataset.originalText ||
                `Run Model ${modelNumber}`;

        }


        if (loading) {

            loading.style.display =
                "none";

        }

    }

}



// ============================================================
// DISPLAY DETECTION RESULTS
// ============================================================

function displayDetectionResults(
    modelNumber,
    data
) {

    const resultContainer =
        document.getElementById(
            `result-${modelNumber}`
        );


    if (!resultContainer) {
        return;
    }


    // Remove old detection information

    const oldInfo =
        resultContainer.querySelector(
            ".detection-info"
        );


    if (oldInfo) {
        oldInfo.remove();
    }


    // No detections

    if (
        !data.detections ||
        !Array.isArray(data.detections) ||
        data.detections.length === 0
    ) {

        return;
    }



    // ========================================================
    // DETECTION INFO CONTAINER
    // ========================================================

    const infoDiv =
        document.createElement(
            "div"
        );

    infoDiv.className =
        "detection-info";



    // ========================================================
    // TITLE
    // ========================================================

    const title =
        document.createElement(
            "h5"
        );

    title.textContent =
        `Detected Objects (${data.detections.length})`;


    infoDiv.appendChild(
        title
    );



    // ========================================================
    // LIST
    // ========================================================

    const list =
        document.createElement(
            "div"
        );

    list.className =
        "detection-list";



    // ========================================================
    // EACH DETECTION
    // ========================================================

    data.detections.forEach(
        function (detection, index) {

            const item =
                document.createElement(
                    "div"
                );

            item.className =
                "detection-item";


            const name =
                detection.name ||
                detection.class ||
                detection.label ||
                `Object ${index + 1}`;


            let confidence = "";


            if (
                detection.confidence !==
                undefined
            ) {

                confidence =
                    `${(
                        Number(
                            detection.confidence
                        ) * 100
                    ).toFixed(1)}%`;

            }


            item.innerHTML = `
                <span class="detection-name">
                    ${escapeHTML(
                        String(name)
                    )}
                </span>

                ${
                    confidence
                    ?
                    `
                    <span class="confidence">
                        ${confidence}
                    </span>
                    `
                    :
                    ""
                }
            `;


            list.appendChild(
                item
            );

        }
    );



    infoDiv.appendChild(
        list
    );


    resultContainer.appendChild(
        infoDiv
    );

}



// ============================================================
// NOTIFICATION
// ============================================================

function showNotification(
    message,
    type = "info"
) {

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
        document.createElement(
            "div"
        );


    notification.className =
        `medrx-notification ${type}`;


    notification.innerHTML = `
        <span class="notification-message">
            ${escapeHTML(
                String(message)
            )}
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


    if (closeButton) {

        closeButton.addEventListener(
            "click",
            function () {

                notification.remove();

            }
        );

    }



    // Auto remove after 5 seconds

    setTimeout(
        function () {

            if (
                document.body.contains(
                    notification
                )
            ) {

                notification.remove();

            }

        },
        5000
    );

}



// ============================================================
// ESCAPE HTML
// ============================================================

function escapeHTML(value) {

    const div =
        document.createElement(
            "div"
        );

    div.textContent =
        value;

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
            link.getAttribute(
                "href"
            );


        if (
            !targetId ||
            targetId === "#"
        ) {

            return;

        }


        const target =
            document.querySelector(
                targetId
            );


        if (target) {

            event.preventDefault();


            target.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        }

    }
);
