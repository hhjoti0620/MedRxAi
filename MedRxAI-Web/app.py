# ============================================================
# MedRxAI WEB APPLICATION
# Prescription OCR + DGDA Medicine Matching
# ============================================================

import os
import re
import uuid
import cv2
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from ultralytics import YOLO
from paddleocr import PaddleOCR


# ============================================================
# 1. FLASK CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
DATABASE_FOLDER = os.path.join(BASE_DIR, "database")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)


app = Flask(__name__)

app.secret_key = "medrxai-development-secret-key"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


# ============================================================
# 2. MODEL PATHS
# ============================================================

MODEL_1_PATH = os.path.join(
    MODEL_FOLDER,
    "best_detector_768_tuned.pt"
)

MODEL_2_PATH = os.path.join(
    MODEL_FOLDER,
    "best_rx_detector_tuned.pt"
)


# ============================================================
# 3. DATABASE PATH
# ============================================================

CSV_PATH = os.path.join(
    DATABASE_FOLDER,
    "medicine.csv"
)


# ============================================================
# 4. LOAD MODELS
# ============================================================

print("\n" + "=" * 70)
print("MEDRxAI INITIALIZATION")
print("=" * 70)


models = {}


def load_models():

    global models

    print("\nLoading YOLO models...")

    if os.path.exists(MODEL_1_PATH):
        try:
            models["model1"] = YOLO(MODEL_1_PATH)
            print("✓ Model 1 loaded")
        except Exception as e:
            print("✗ Model 1 loading failed:", e)
    else:
        print("⚠ Model 1 not found:", MODEL_1_PATH)

    if os.path.exists(MODEL_2_PATH):
        try:
            models["model2"] = YOLO(MODEL_2_PATH)
            print("✓ Model 2 loaded")
        except Exception as e:
            print("✗ Model 2 loading failed:", e)
    else:
        print("⚠ Model 2 not found:", MODEL_2_PATH)


load_models()


# ============================================================
# 5. LOAD PADDLE OCR
# ============================================================

print("\nLoading PaddleOCR...")

try:

    ocr_engine = PaddleOCR(lang="en")

    print("✓ PaddleOCR loaded")

except Exception as e:

    print("✗ PaddleOCR initialization failed:")
    print(e)

    ocr_engine = None


# ============================================================
# 6. LOAD DGDA MEDICINE DATABASE
# ============================================================

df_dgda = None

unique_brands = []
meta_lookup = {}
generic_to_brands = {}


def load_database():

    global df_dgda
    global unique_brands
    global meta_lookup
    global generic_to_brands

    print("\nLoading medicine database...")

    if not os.path.exists(CSV_PATH):

        print("⚠ medicine.csv not found:")
        print(CSV_PATH)

        return False

    try:

        raw_df = pd.read_csv(
            CSV_PATH,
            low_memory=False
        )

        col_map = {
            c.strip().lower(): c
            for c in raw_df.columns
        }

        # ----------------------------------------------------
        # Brand
        # ----------------------------------------------------

        brand_col = next(
            (
                col_map[c]
                for c in [
                    "brand_name",
                    "brand name",
                    "brand",
                    "name",
                    "medicine_name"
                ]
                if c in col_map
            ),
            None
        )

        # ----------------------------------------------------
        # Generic
        # ----------------------------------------------------

        generic_col = next(
            (
                col_map[c]
                for c in [
                    "generic_name",
                    "generic name",
                    "generic"
                ]
                if c in col_map
            ),
            None
        )

        # ----------------------------------------------------
        # Strength
        # ----------------------------------------------------

        strength_col = next(
            (
                col_map[c]
                for c in [
                    "strength",
                    "dosage",
                    "formulation"
                ]
                if c in col_map
            ),
            None
        )

        # ----------------------------------------------------
        # Manufacturer
        # ----------------------------------------------------

        mfg_col = next(
            (
                col_map[c]
                for c in [
                    "manufacturer",
                    "company_name",
                    "company",
                    "mfg"
                ]
                if c in col_map
            ),
            None
        )

        if brand_col is None:

            raise ValueError(
                "Brand column was not found in medicine.csv"
            )

        if generic_col is None:

            raise ValueError(
                "Generic column was not found in medicine.csv"
            )

        if strength_col is None:

            raise ValueError(
                "Strength column was not found in medicine.csv"
            )

        # ----------------------------------------------------
        # Select columns
        # ----------------------------------------------------

        df_cols = [
            brand_col,
            generic_col,
            strength_col
        ]

        if mfg_col:
            df_cols.append(mfg_col)

        df_dgda = raw_df[df_cols].copy()

        # ----------------------------------------------------
        # Rename
        # ----------------------------------------------------

        rename_dict = {

            brand_col: "brand",

            generic_col: "generic",

            strength_col: "strength"
        }

        if mfg_col:
            rename_dict[mfg_col] = "manufacturer"

        df_dgda.rename(
            columns=rename_dict,
            inplace=True
        )

        # ----------------------------------------------------
        # Manufacturer fallback
        # ----------------------------------------------------

        if "manufacturer" not in df_dgda.columns:

            df_dgda["manufacturer"] = "N/A"

        # ----------------------------------------------------
        # Clean
        # ----------------------------------------------------

        df_dgda.dropna(
            subset=["brand"],
            inplace=True
        )

        for col in [
            "brand",
            "generic",
            "strength",
            "manufacturer"
        ]:

            df_dgda[col] = (
                df_dgda[col]
                .astype(str)
                .str.strip()
            )

        # ----------------------------------------------------
        # Deduplicate
        # ----------------------------------------------------

        df_dgda.drop_duplicates(
            subset=[
                "brand",
                "manufacturer"
            ],
            inplace=True
        )

        # ----------------------------------------------------
        # Create indexes
        # ----------------------------------------------------

        unique_brands = (
            df_dgda["brand"]
            .unique()
            .tolist()
        )

        meta_lookup = (
            df_dgda
            .drop_duplicates(
                subset=["brand"]
            )
            .set_index("brand")
            [
                [
                    "generic",
                    "strength",
                    "manufacturer"
                ]
            ]
            .to_dict(
                orient="index"
            )
        )

        generic_to_brands = (
            df_dgda
            .groupby("generic")["brand"]
            .apply(
                lambda s:
                sorted(
                    list(
                        set(s)
                    )
                )
            )
            .to_dict()
        )

        print(
            f"✓ Medicine records: {len(df_dgda):,}"
        )

        print(
            f"✓ Unique brands: {len(unique_brands):,}"
        )

        return True

    except Exception as e:

        print(
            "✗ Database loading failed:"
        )

        print(e)

        return False


database_loaded = load_database()


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# THERAPEUTIC CATEGORY
# ============================================================

def get_therapeutic_category(generic_name):

    if not isinstance(
        generic_name,
        str
    ):
        return "Other"

    g = generic_name.lower()

    if any(
        k in g
        for k in [
            "vitamin",
            "mineral",
            "antioxidant",
            "multivitamin",
            "calcium",
            "iron",
            "acid"
        ]
    ):

        return "Supplement"

    elif any(
        k in g
        for k in [
            "prazole",
            "antacid",
            "ranitidine",
            "esomeprazole",
            "omeprazole"
        ]
    ):

        return "Gastro / PPI"

    elif any(
        k in g
        for k in [
            "domperidone",
            "ondansetron",
            "levosulpiride"
        ]
    ):

        return "Antiemetic"

    elif any(
        k in g
        for k in [
            "ursodeoxycholic",
            "silymarin",
            "liver"
        ]
    ):

        return "Liver / Digestion"

    return "Other"


# ============================================================
# LEVENSHTEIN
# ============================================================

def levenshtein_dist(
    s1,
    s2
):

    s1 = str(s1).lower().strip()
    s2 = str(s2).lower().strip()

    dp = [
        [0] * (len(s2) + 1)
        for _ in range(len(s1) + 1)
    ]

    for i in range(
        len(s1) + 1
    ):

        dp[i][0] = i

    for j in range(
        len(s2) + 1
    ):

        dp[0][j] = j

    for i in range(
        1,
        len(s1) + 1
    ):

        for j in range(
            1,
            len(s2) + 1
        ):

            if s1[i - 1] == s2[j - 1]:

                dp[i][j] = (
                    dp[i - 1][j - 1]
                )

            else:

                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1]
                )

    return dp[-1][-1]


def similarity_ratio(
    s1,
    s2
):

    d = levenshtein_dist(
        s1,
        s2
    )

    m = max(
        len(str(s1)),
        len(str(s2))
    )

    if m == 0:

        return 100.0

    return (
        1.0 -
        (d / m)
    ) * 100.0


# ============================================================
# OCR VARIANTS
# ============================================================

def generate_dynamic_ocr_variants(
    token
):

    variants = {token}

    low = token.lower()

    substitutions = [

        ("y", "r"),
        ("r", "y"),

        ("v", "r"),
        ("r", "v"),

        ("i", "l"),
        ("l", "i"),

        ("i", "t"),
        ("t", "i"),

        ("c", "e"),
        ("e", "c"),

        ("c", "o"),
        ("o", "c"),

        ("n", "m"),
        ("m", "n"),

        ("a", "o"),
        ("o", "a")
    ]

    for char1, char2 in substitutions:

        if char1 in low:

            variants.add(
                token.replace(
                    char1,
                    char2
                ).replace(
                    char1.upper(),
                    char2.upper()
                )
            )

    return list(variants)


# ============================================================
# 8. PRESCRIPTION ANALYSIS
# ============================================================

def analyze_prescription(
    image_path,
    model_key
):

    if model_key not in models:

        raise RuntimeError(
            f"{model_key} is not available."
        )

    if not database_loaded:

        raise RuntimeError(
            "medicine.csv could not be loaded."
        )

    if ocr_engine is None:

        raise RuntimeError(
            "PaddleOCR is not available."
        )

    detector = models[model_key]

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    raw_bgr = cv2.imread(
        image_path
    )

    if raw_bgr is None:

        raise ValueError(
            "Could not read uploaded image."
        )

    orig_h, orig_w = raw_bgr.shape[:2]

    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    preds = detector.predict(

        image_path,

        imgsz=768,

        conf=0.40,

        iou=0.40,

        verbose=False

    )[0]

    boxes = (
        preds.boxes.xyxy
        .cpu()
        .numpy()
    )

    confs = (
        preds.boxes.conf
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Sort top-to-bottom
    # --------------------------------------------------------

    if len(boxes) > 0:

        sort_idx = np.argsort(
            [
                b[1]
                for b in boxes
            ]
        )

        boxes = [
            boxes[i]
            for i in sort_idx
        ]

        confs = [
            confs[i]
            for i in sort_idx
        ]

    # --------------------------------------------------------
    # Empty detection
    # --------------------------------------------------------

    if len(boxes) == 0:

        return {

            "success": False,

            "message":
            "No medicine region was detected.",

            "detected_count": 0,

            "clinical_report": [],

            "alternatives_report": [],

            "annotated_image": None
        }

    live_extractions = []

    crops_enh = []

    # ========================================================
    # OCR
    # ========================================================

    for idx, (
        box,
        det_c
    ) in enumerate(
        zip(
            boxes,
            confs
        )
    ):

        x1, y1, x2, y2 = map(
            int,
            box
        )

        bw = x2 - x1
        bh = y2 - y1

        pad_w = int(
            bw * 0.06
        )

        pad_h = int(
            bh * 0.06
        )

        px1 = max(
            0,
            x1 - pad_w
        )

        py1 = max(
            0,
            y1 - pad_h
        )

        px2 = min(
            orig_w,
            x2 + pad_w
        )

        py2 = min(
            orig_h,
            y2 + pad_h
        )

        crop_r = raw_bgr[
            py1:py2,
            px1:px2
        ]

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        crop_g = cv2.cvtColor(
            crop_r,
            cv2.COLOR_BGR2GRAY
        )

        denoised = cv2.bilateralFilter(
            crop_g,
            d=5,
            sigmaColor=40,
            sigmaSpace=40
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.2,
            tileGridSize=(8, 8)
        )

        crop_e = clahe.apply(
            denoised
        )

        crop_e_bgr = cv2.cvtColor(
            crop_e,
            cv2.COLOR_GRAY2BGR
        )

        token = "N/A"
        ocr_c = 0.0

        # ----------------------------------------------------
        # PaddleOCR
        # ----------------------------------------------------

        try:

            res = ocr_engine.predict(

                crop_e_bgr,

                use_doc_orientation_classify=False,

                use_doc_unwarping=False,

                use_textline_orientation=False
            )

            if res and len(res) > 0:

                texts = res[0].get(
                    "rec_texts",
                    []
                )

                scores = res[0].get(
                    "rec_scores",
                    []
                )

                if len(texts) > 0:

                    token = " ".join(
                        [
                            str(t).strip()
                            for t in texts
                            if str(t).strip()
                        ]
                    )

                    if len(scores) > 0:

                        ocr_c = float(
                            np.mean(scores)
                        )

        except Exception as e:

            print(
                "OCR error:",
                e
            )

        # ----------------------------------------------------
        # Skip dosage-only lines
        # ----------------------------------------------------

        if re.search(
            r"\d",
            token
        ):

            continue

        crops_enh.append(
            crop_e
        )

        live_extractions.append({

            "roi": idx + 1,

            "box": (
                x1,
                y1,
                x2,
                y2
            ),

            "raw_token": token,

            "det_conf": float(
                det_c
            ),

            "ocr_conf": float(
                ocr_c
            )
        })

    # ========================================================
    # ERROR-TOLERANT MATCHING
    # ========================================================

    clinical_stopwords = {

        "cap",
        "caps",
        "capsule",

        "tab",
        "tabs",
        "tablet",

        "inj",

        "syp",
        "syrup",

        "drop",

        "mg",
        "ml",
        "gm"
    }

    # --------------------------------------------------------
    # Prescription context
    # --------------------------------------------------------

    prescription_context = []

    for item in live_extractions:

        tok = item["raw_token"]

        clean_tok = re.sub(
            r"[^a-zA-Z\s-]",
            "",
            tok
        ).strip()

        if len(clean_tok) >= 4:

            best_b = None
            max_s = -1.0

            for brand in unique_brands:

                sim = similarity_ratio(
                    clean_tok,
                    brand
                )

                if sim > max_s:

                    max_s = sim
                    best_b = brand

            if (
                max_s >= 80.0
                and best_b
            ):

                meta = meta_lookup.get(
                    best_b,
                    {}
                )

                cat = get_therapeutic_category(
                    meta.get(
                        "generic",
                        ""
                    )
                )

                prescription_context.append(
                    cat
                )

    clinical_report = []
    alternatives_report = []

    # --------------------------------------------------------
    # Match every OCR token
    # --------------------------------------------------------

    for item in live_extractions:

        tok = item["raw_token"]

        clean_tok = re.sub(
            r"[^a-zA-Z\s-]",
            "",
            tok
        ).strip()

        # Invalid token
        if (
            clean_tok.lower()
            in clinical_stopwords
            or len(clean_tok) < 2
        ):

            clinical_report.append({

                "roi":
                f"Med #{item['roi']}",

                "raw_ocr":
                tok,

                "ocr_conf":
                round(
                    item["ocr_conf"] * 100,
                    1
                ),

                "brand":
                "N/A",

                "generic":
                "N/A",

                "strength":
                "N/A",

                "manufacturer":
                "N/A",

                "similarity":
                0.0,

                "status":
                "Rejected: Invalid Token",

                "category":
                "N/A"
            })

            continue

        token_variants = (
            generate_dynamic_ocr_variants(
                clean_tok
            )
        )

        best_match = None
        max_sim = -1.0
        final_matched_variant = clean_tok

        # ----------------------------------------------------
        # Compare against DGDA brands
        # ----------------------------------------------------

        for var in token_variants:

            candidate_brands = unique_brands

            if len(var) <= 3:

                candidate_brands = [

                    b
                    for b in unique_brands

                    if str(b)
                    .lower()
                    .startswith(
                        var[0].lower()
                    )
                ]

            for brand in candidate_brands:

                if abs(
                    len(var)
                    -
                    len(brand)
                ) > 4:

                    continue

                sim = similarity_ratio(
                    var,
                    brand
                )

                meta_test = meta_lookup.get(
                    brand,
                    {}
                )

                brand_cat = (
                    get_therapeutic_category(
                        meta_test.get(
                            "generic",
                            ""
                        )
                    )
                )

                if (
                    brand_cat
                    in prescription_context
                    and len(var) <= 3
                ):

                    sim += 12.0

                if sim > max_sim:

                    max_sim = sim

                    best_match = brand

                    final_matched_variant = var

                    if sim >= 100.0:
                        break

        # ----------------------------------------------------
        # Successful match
        # ----------------------------------------------------

        if (
            max_sim >= 50.0
            and best_match
        ):

            meta = meta_lookup.get(
                best_match,
                {}
            )

            gen = meta.get(
                "generic",
                "N/A"
            )

            strength = meta.get(
                "strength",
                "N/A"
            )

            manufacturer = meta.get(
                "manufacturer",
                "N/A"
            )

            alternatives = [

                b
                for b
                in generic_to_brands.get(
                    gen,
                    []
                )

                if str(b).lower()
                !=
                str(best_match).lower()
            ]

            category = get_therapeutic_category(
                gen
            )

            corrected = (

                final_matched_variant
                if
                final_matched_variant.lower()
                !=
                clean_tok.lower()

                else
                None
            )

            clinical_report.append({

                "roi":
                f"Med #{item['roi']}",

                "raw_ocr":
                tok,

                "corrected":
                corrected,

                "ocr_conf":
                round(
                    item["ocr_conf"] * 100,
                    1
                ),

                "brand":
                best_match,

                "generic":
                gen,

                "strength":
                strength,

                "manufacturer":
                manufacturer,

                "similarity":
                round(
                    min(
                        max_sim,
                        100.0
                    ),
                    1
                ),

                "status":
                "Confirmed Match",

                "category":
                category
            })

            alternatives_report.append({

                "brand":
                best_match,

                "generic":
                gen,

                "alternatives":
                alternatives[:5]
            })

        # ----------------------------------------------------
        # Failed match
        # ----------------------------------------------------

        else:

            clinical_report.append({

                "roi":
                f"Med #{item['roi']}",

                "raw_ocr":
                tok,

                "corrected":
                None,

                "ocr_conf":
                round(
                    item["ocr_conf"] * 100,
                    1
                ),

                "brand":
                "N/A",

                "generic":
                "N/A",

                "strength":
                "N/A",

                "manufacturer":
                "N/A",

                "similarity":
                round(
                    max_sim
                    if max_sim > 0
                    else 0.0,
                    1
                ),

                "status":
                "Possible/Low Confidence",

                "category":
                "N/A"
            })

    # ========================================================
    # ANNOTATED IMAGE
    # ========================================================

    annotated = raw_bgr.copy()

    for item in live_extractions:

        bx1, by1, bx2, by2 = item["box"]

        cv2.rectangle(

            annotated,

            (bx1, by1),

            (bx2, by2),

            (0, 180, 0),

            2
        )

        label = (
            f"#{item['roi']} "
            f"{item['raw_token']}"
        )

        cv2.putText(

            annotated,

            label,

            (
                bx1,
                max(
                    20,
                    by1 - 8
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (0, 180, 0),

            2
        )

    # --------------------------------------------------------
    # Save annotated image
    # --------------------------------------------------------

    result_filename = (
        "annotated_"
        + uuid.uuid4().hex
        + ".jpg"
    )

    result_path = os.path.join(
        RESULT_FOLDER,
        result_filename
    )

    cv2.imwrite(
        result_path,
        annotated
    )

    return {

        "success": True,

        "message":
        "Prescription analyzed successfully.",

        "detected_count":
        len(live_extractions),

        "clinical_report":
        clinical_report,

        "alternatives_report":
        alternatives_report,

        "annotated_image":
        result_filename
    }


# ============================================================
# 9. HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def index():

    model_status = {

        "model1":
        "model1" in models,

        "model2":
        "model2" in models,

        "database":
        database_loaded,

        "ocr":
        ocr_engine is not None
    }

    return render_template(
        "index.html",
        model_status=model_status,
        result=None
    )


# ============================================================
# 10. ANALYZE ROUTE
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if "prescription" not in request.files:

        flash(
            "Please select a prescription image."
        )

        return redirect(
            url_for("index")
        )

    file = request.files[
        "prescription"
    ]

    if file.filename == "":

        flash(
            "No image selected."
        )

        return redirect(
            url_for("index")
        )

    if not allowed_file(
        file.filename
    ):

        flash(
            "Only JPG, JPEG, PNG and WEBP images are supported."
        )

        return redirect(
            url_for("index")
        )

    # --------------------------------------------------------
    # Model selection
    # --------------------------------------------------------

    selected_model = request.form.get(
        "model",
        "model1"
    )

    if selected_model not in models:

        flash(
            "Selected model is not available."
        )

        return redirect(
            url_for("index")
        )

    # --------------------------------------------------------
    # Save uploaded file
    # --------------------------------------------------------

    original_name = secure_filename(
        file.filename
    )

    unique_name = (
        uuid.uuid4().hex
        + "_"
        + original_name
    )

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )

    file.save(
        upload_path
    )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    try:

        result = analyze_prescription(
            upload_path,
            selected_model
        )

        result["model_name"] = (
            "Model 1"
            if selected_model == "model1"
            else "Model 2"
        )

        result["uploaded_image"] = (
            unique_name
        )

        return render_template(
            "index.html",

            model_status={
                "model1":
                "model1" in models,

                "model2":
                "model2" in models,

                "database":
                database_loaded,

                "ocr":
                ocr_engine is not None
            },

            result=result
        )

    except Exception as e:

        print(
            "Analysis Error:",
            e
        )

        flash(
            f"Analysis failed: {str(e)}"
        )

        return redirect(
            url_for("index")
        )


# ============================================================
# 11. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)

    print(
        "MEDRxAI WEB SERVER"
    )

    print("=" * 70)

    print(
        "Open: http://127.0.0.1:5000"
    )

    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
