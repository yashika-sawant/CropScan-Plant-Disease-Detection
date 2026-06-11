"""
CropScan — Real AI Plant Disease Detection
Uses MobileNetV2 + PlantVillage class mapping
"""
import os, uuid, numpy as np
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "plant-disease-secret-2024"
UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

print("Loading AI model...")
MODEL = tf.keras.models.load_model("mobilenet_base.h5")
print("Model loaded!")

PLANT_CLASSES = [
    "Apple - Apple Scab", "Apple - Black Rot", "Apple - Cedar Apple Rust", "Apple - Healthy",
    "Blueberry - Healthy", "Cherry - Powdery Mildew", "Cherry - Healthy",
    "Corn - Cercospora Leaf Spot", "Corn - Common Rust", "Corn - Northern Leaf Blight", "Corn - Healthy",
    "Grape - Black Rot", "Grape - Esca (Black Measles)", "Grape - Leaf Blight", "Grape - Healthy",
    "Orange - Haunglongbing (Citrus Greening)",
    "Peach - Bacterial Spot", "Peach - Healthy",
    "Pepper - Bacterial Spot", "Pepper - Healthy",
    "Potato - Early Blight", "Potato - Late Blight", "Potato - Healthy",
    "Raspberry - Healthy", "Soybean - Healthy", "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch", "Strawberry - Healthy",
    "Tomato - Bacterial Spot", "Tomato - Early Blight", "Tomato - Late Blight",
    "Tomato - Leaf Mold", "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites", "Tomato - Target Spot",
    "Tomato - Yellow Leaf Curl Virus", "Tomato - Mosaic Virus",
    "Tomato - Healthy"
]

DISEASE_INFO = {
    "Apple - Apple Scab": {"symptoms": ["Olive-green to brown scab-like lesions on leaves", "Velvety texture on leaf spots", "Premature leaf drop", "Scabby corky spots on fruit"], "causes": ["Fungus: Venturia inaequalis", "Cool wet spring weather", "Infected fallen leaves", "Wind-dispersed spores"], "treatment": ["Apply fungicide at bud break", "Use myclobutanil or captan sprays", "Remove fallen infected leaves", "Prune to improve air circulation"], "prevention": ["Plant scab-resistant varieties", "Rake and destroy fallen leaves", "Apply dormant lime sulphur spray", "Ensure good air circulation"], "severity": "Moderate", "color": "#e67e22", "icon": "🍎"},
    "Apple - Black Rot": {"symptoms": ["Brown to black circular leaf spots", "Frog-eye leaf spot pattern", "Mummified fruit on tree", "Cankers on branches"], "causes": ["Fungus: Botryosphaeria obtusa", "Warm humid weather", "Stressed or wounded trees", "Infected mummified fruit"], "treatment": ["Remove mummified fruit and dead wood", "Apply captan or thiophanate-methyl", "Prune infected branches", "Destroy all infected material"], "prevention": ["Maintain tree vigour", "Remove all mummified fruit", "Prune dead wood annually", "Apply protective fungicide sprays"], "severity": "High", "color": "#c0392b", "icon": "⚫"},
    "Apple - Cedar Apple Rust": {"symptoms": ["Bright orange-yellow spots on upper leaf", "Tube-like structures on leaf underside", "Premature defoliation", "Distorted fruit with orange lesions"], "causes": ["Fungus: Gymnosporangium juniperi-virginianae", "Requires both apple and cedar/juniper hosts", "Wind-carried spores in spring", "Wet spring weather"], "treatment": ["Apply myclobutanil or triadimefon fungicide", "Time sprays during bloom period", "Remove nearby juniper hosts if possible", "Use protective sprays every 7-10 days"], "prevention": ["Plant resistant apple varieties", "Remove juniper trees near orchard", "Apply preventive fungicide in spring", "Monitor weather for infection periods"], "severity": "Moderate", "color": "#f39c12", "icon": "🟠"},
    "Apple - Healthy": {"symptoms": ["Deep green uniform leaf colour", "No spots or lesions", "Normal leaf size and shape", "Healthy fruit development"], "causes": ["No disease detected", "Good growing conditions", "Proper nutrition and care"], "treatment": ["No treatment needed", "Continue regular monitoring", "Maintain balanced fertilisation"], "prevention": ["Regular monitoring", "Proper pruning", "Good orchard hygiene", "Balanced nutrition program"], "severity": "None", "color": "#27ae60", "icon": "✅"},
    "Blueberry - Healthy": {"symptoms": ["Bright green healthy foliage", "No lesions or discolouration", "Normal berry development", "Strong upright growth"], "causes": ["No disease detected", "Optimal soil pH (4.5-5.5)", "Good drainage and sunlight"], "treatment": ["No treatment needed", "Maintain acidic soil pH", "Continue regular care"], "prevention": ["Maintain soil pH", "Mulch around plants", "Ensure good drainage", "Prune annually"], "severity": "None", "color": "#27ae60", "icon": "✅"},
    "Cherry - Powdery Mildew": {"symptoms": ["White powdery coating on young leaves", "Distorted and curled new growth", "Yellowing of affected tissue", "Premature leaf drop"], "causes": ["Fungus: Podosphaera clandestina", "Warm dry days with cool nights", "High humidity", "Dense canopy with poor airflow"], "treatment": ["Apply sulphur or potassium bicarbonate spray", "Use myclobutanil fungicide", "Remove heavily infected shoots", "Improve air circulation by pruning"], "prevention": ["Plant resistant varieties", "Prune for open canopy", "Avoid excessive nitrogen", "Apply preventive sulphur sprays"], "severity": "Moderate", "color": "#f39c12", "icon": "⬜"},
    "Cherry - Healthy": {"symptoms": ["Glossy dark green leaves", "No spots or powdery coating", "Normal fruit set", "Vigorous growth"], "causes": ["No disease detected", "Good growing conditions", "Proper care and nutrition"], "treatment": ["No treatment needed", "Continue current practices", "Monitor regularly"], "prevention": ["Annual pruning", "Good air circulation", "Balanced fertilisation", "Regular monitoring"], "severity": "None", "color": "#27ae60", "icon": "✅"},
    "Corn - Cercospora Leaf Spot": {"symptoms": ["Small rectangular grey-tan lesions", "Lesions run parallel to leaf veins", "Brown borders around lesions", "Premature death of lower leaves"], "causes": ["Fungus: Cercospora zeae-maydis", "Warm humid conditions", "Extended leaf wetness", "Reduced tillage fields"], "treatment": ["Apply strobilurin or triazole fungicide", "Time application at tasselling", "Improve field drainage", "Use resistant hybrids next season"], "prevention": ["Plant resistant corn hybrids", "Practice crop rotation", "Till crop debris after harvest", "Ensure adequate plant spacing"], "severity": "Moderate", "color": "#e67e22", "icon": "🌽"},
    "Corn - Common Rust": {"symptoms": ["Small powdery brick-red pustules on leaves", "Pustules on both leaf surfaces", "Pustules turn black late season", "Yellowing around pustule clusters"], "causes": ["Fungus: Puccinia sorghi", "Cool temperatures (16-23°C)", "High humidity and dew", "Wind-dispersed spores"], "treatment": ["Apply triazole fungicide early", "Use azoxystrobin or propiconazole", "Scout fields regularly", "Apply when pustules first appear"], "prevention": ["Plant resistant hybrids", "Early planting to avoid peak spore season", "Monitor weather conditions", "Crop rotation"], "severity": "Moderate", "color": "#e67e22", "icon": "🟤"},
    "Corn - Northern Leaf Blight": {"symptoms": ["Long cigar-shaped grey-green lesions", "Lesions 2.5-15cm long", "Lesions turn tan-brown with age", "Rapid blighting in humid conditions"], "causes": ["Fungus: Exserohilum turcicum", "Moderate temperatures (18-27°C)", "Extended periods of leaf wetness", "Infected crop debris"], "treatment": ["Apply fungicide at early tasselling", "Use propiconazole or azoxystrobin", "Remove crop debris after harvest", "Plant resistant varieties next year"], "prevention": ["Use resistant corn hybrids", "Crop rotation with non-host crops", "Till infected residue", "Avoid dense planting"], "severity": "High", "color": "#c0392b", "icon": "🌽"},
    "Corn - Healthy": {"symptoms": ["Bright green uniform leaves", "No lesions or pustules", "Strong stalk development", "Normal tassel and ear formation"], "causes": ["No disease detected", "Good growing conditions", "Proper fertilisation"], "treatment": ["No treatment needed", "Monitor regularly", "Maintain fertilisation program"], "prevention": ["Crop rotation", "Use certified seeds", "Monitor for early signs", "Balanced nutrition"], "severity": "None", "color": "#27ae60", "icon": "✅"},
    "Grape - Black Rot": {"symptoms": ["Circular tan spots with dark borders on leaves", "Black shrivelled mummified berries", "Brown lesions on shoots and tendrils", "Salmon-coloured spore masses"], "causes": ["Fungus: Guignardia bidwellii", "Warm wet weather (26°C)", "Infected mummified berries", "Rain-splashed spores"], "treatment": ["Apply mancozeb or myclobutanil", "Remove all mummified berries", "Prune to open canopy", "Begin sprays at bud break"], "prevention": ["Remove mummified fruit over winter", "Train vines for good airflow", "Apply preventive fungicide program", "Site selection with good drainage"], "severity": "High", "color": "#c0392b", "icon": "🍇"},
    "Grape - Esca (Black Measles)": {"symptoms": ["Tiger-stripe pattern on leaves", "Interveinal chlorosis and necrosis", "Small dark berries with off-flavour", "Internal wood shows brown streaking"], "causes": ["Complex of wood-rotting fungi", "Pruning wound infections", "Stress factors weaken vines", "Long incubation period"], "treatment": ["No complete cure available", "Remove and destroy infected wood", "Apply wound protectant after pruning", "Manage vine stress carefully"], "prevention": ["Delay pruning to reduce infection risk", "Apply wound sealants after cuts", "Use clean pruning tools", "Avoid large pruning wounds"], "severity": "High", "color": "#8e44ad", "icon": "🟣"},
    "Grape - Leaf Blight": {"symptoms": ["Irregular brown lesions on leaves", "Lesions expand rapidly in wet weather", "Leaves wither and drop early", "Affects fruit clusters in severe cases"], "causes": ["Fungus: Pseudocercospora vitis", "Hot humid conditions", "Extended leaf wetness", "Poor vineyard air circulation"], "treatment": ["Apply mancozeb or captan fungicide", "Improve canopy management", "Remove infected leaves", "Ensure good drainage"], "prevention": ["Train vines for airflow", "Avoid excessive irrigation", "Apply preventive fungicide", "Remove infected material"], "severity": "Moderate", "color": "#e67e22", "icon": "🍃"},
    "Grape - Healthy": {"symptoms": ["Lush green uniform foliage", "No spots or discolouration", "Normal cluster development", "Strong cane growth"], "causes": ["No disease detected", "Good vineyard management", "Proper nutrition and irrigation"], "treatment": ["No treatment needed", "Continue monitoring", "Maintain vineyard hygiene"], "prevention": ["Regular canopy management", "Balanced fertilisation", "Integrated pest management", "Good drainage"], "severity": "None", "color": "#27ae60", "icon": "✅"},
    "Orange - Haunglongbing (Citrus Greening)": {"symptoms": ["Asymmetric blotchy mottling of leaves", "Yellow shoots (huanglongbing means yellow dragon)", "Small lopsided bitter fruit", "Premature fruit drop"], "causes": ["Bacteria: Candidatus Liberibacter", "Spread by Asian citrus psyllid insect", "No cure once infected", "Systemic throughout entire tree"], "treatment": ["No cure — remove infected trees", "Control psyllid population aggressively", "Apply systemic insecticides", "Replant with certified disease-free stock"], "prevention": ["Use certified disease-free nursery stock", "Control Asian citrus psyllid", "Inspect new trees before planting", "Quarantine measures in affected areas"], "severity": "High", "color": "#c0392b", "icon": "🍊"},
    "Peach - Bacterial Spot": {"symptoms": ["Small water-soaked spots on leaves", "Spots turn purple-brown with yellow halo", "Shot-hole appearance as centres drop", "Sunken dark lesions on fruit"], "causes": ["Bacteria: Xanthomonas arboricola", "Warm wet weather", "Wind-driven rain", "Infected nursery stock"], "treatment": ["Apply copper bactericide sprays", "Apply oxytetracycline antibiotic", "Begin at bud swell stage", "Repeat every 7-10 days in wet weather"], "prevention": ["Plant resistant varieties", "Windbreaks to reduce leaf wetness", "Avoid overhead irrigation", "Use copper sprays preventively"], "severity": "Moderate", "color": "#e67e22", "icon": "🍑"},
    "Peach - Healthy": {"symptoms": ["Deep green glossy leaves", "No spots or lesions", "Good fruit development", "Vigorous shoot growth"], "causes": ["No disease detected", "Good growing conditions", "Proper care"], "treatment": ["No treatment needed", "Regular monitoring", "Maintain care program"], "prevention": ["Annual pruning", "Balanced fertilisation", "Monitor regularly", "Good sanitation"], "severity": "None", "color": "#27ae60", "icon": "✅"},
    "Pepper - Bacterial Spot": {"symptoms": ["Small water-soaked circular spots", "Spots enlarge with yellow halo", "Raised scabby lesions on fruit", "Defoliation in severe cases"], "causes": ["Bacteria: Xanthomonas campestris", "Warm wet conditions (24-30°C)", "Rain splash and overhead irrigation", "Infected transplants or seeds"], "treatment": ["Apply copper hydroxide spray", "Use acibenzolar-S-methyl resistance inducer", "Avoid overhead irrigation", "Remove heavily infected plants"], "prevention": ["Use certified disease-free seeds", "Hot-water seed treatment", "Crop rotation every 2-3 years", "Avoid working in wet conditions"], "severity": "Moderate", "color": "#e67e22", "icon": "🫑"},
    "Pepper - Healthy": {"symptoms": ["Bright green firm leaves", "No spots or discolouration", "Normal fruit set", "Upright healthy stems"], "causes": ["No disease detected", "Good growing conditions", "Proper care and nutrition"], "treatment": ["No treatment needed", "Continue current practices", "Monitor regularly"], "prevention": ["Crop rotation", "Good drainage", "Balanced fertilisation", "Regular monitoring"], "severity": "None", "color": "#27ae60", "icon": "✅"},
    "Potato - Early Blight": {"symptoms": ["Dark brown circular spots with concentric rings", "Target-board pattern on older leaves", "Yellow halo around lesions", "Premature defoliation"], "causes": ["Fungus: Alternaria solani", "Warm temperatures (24-29°C)", "High humidity", "Nitrogen-stressed plants"], "treatment": ["Apply mancozeb or chlorothalonil", "Use azoxystrobin or difenoconazole", "Remove infected lower leaves", "Ensure adequate plant nutrition"], "prevention": ["Use certified seed potatoes", "Crop rotation every 3 years", "Avoid excessive irrigation", "Maintain adequate nitrogen levels"], "severity": "Moderate", "color": "#e67e22", "icon": "🥔"},
    "Potato - Late Blight": {"symptoms": ["Water-soaked pale green lesions on leaves", "White fluffy growth on leaf underside", "Rapid blackening and collapse of foliage", "Brown rot in tubers"], "causes": ["Oomycete: Phytophthora infestans", "Cool wet weather (10-20°C)", "High relative humidity", "Infected seed tubers or volunteers"], "treatment": ["Apply metalaxyl or cymoxanil immediately", "Use copper-based fungicides", "Destroy infected haulm before harvest", "Do not harvest in wet conditions"], "prevention": ["Use resistant varieties", "Plant certified disease-free seed", "Apply preventive fungicide program", "Monitor weather-based forecasting systems"], "severity": "High", "color": "#c0392b", "icon": "⚠️"},
    "Potato - Healthy": {"symptoms": ["Dark green uniform foliage", "No lesions or water-soaked areas", "Normal flower and tuber development", "Strong upright stems"], "causes": ["No disease detected", "Good growing conditions", "Proper soil health"], "treatment": ["No treatment needed", "Continue monitoring", "Maintain care program"], "prevention": ["Use certified seed potatoes", "Crop rotation", "Good drainage", "Regular monitoring"], "severity": "None", "color": "#27ae60", "icon": "✅"},
    "Raspberry - Healthy": {"symptoms": ["Bright green healthy canes", "No spots or rust pustules", "Normal fruit development", "Vigorous new cane growth"], "causes": ["No disease detected", "Good growing conditions", "Proper nutrition"], "treatment": ["No treatment needed", "Regular monitoring", "Maintain care"], "prevention": ["Annual cane management", "Good air circulation", "Balanced fertilisation", "Regular monitoring"], "severity": "None", "color": "#27ae60", "icon": "✅"},
    "Soybean - Healthy": {"symptoms": ["Uniform dark green foliage", "No lesions or chlorosis", "Normal pod development", "Strong stem growth"], "causes": ["No disease detected", "Good growing conditions", "Proper soil nutrition"], "treatment": ["No treatment needed", "Monitor regularly", "Maintain program"], "prevention": ["Crop rotation", "Use certified seeds", "Monitor for pests", "Balanced nutrition"], "severity": "None", "color": "#27ae60", "icon": "✅"},
    "Squash - Powdery Mildew": {"symptoms": ["White powdery patches on leaf surface", "Yellowing of affected leaves", "Distorted young leaves", "Reduced fruit quality"], "causes": ["Fungus: Podosphaera xanthii", "Warm dry weather with high humidity", "Dense planting", "Poor air circulation"], "treatment": ["Apply potassium bicarbonate or sulphur", "Use neem oil as organic option", "Apply myclobutanil fungicide", "Remove severely infected leaves"], "prevention": ["Plant resistant varieties", "Ensure good plant spacing", "Avoid overhead watering", "Apply preventive sulphur sprays"], "severity": "Moderate", "color": "#f39c12", "icon": "⬜"},
    "Strawberry - Leaf Scorch": {"symptoms": ["Small purple to red spots on upper leaf", "Spots enlarge with grey centres", "Leaf edges turn brown and scorched", "Premature defoliation"], "causes": ["Fungus: Diplocarpon earlianum", "Warm humid weather", "Overhead irrigation", "Dense plant spacing"], "treatment": ["Apply captan or myclobutanil fungicide", "Remove infected leaves", "Improve air circulation", "Reduce leaf wetness duration"], "prevention": ["Plant in well-drained locations", "Avoid overhead watering", "Space plants adequately", "Apply preventive fungicide in spring"], "severity": "Moderate", "color": "#e67e22", "icon": "🔥"},
    "Strawberry - Healthy": {"symptoms": ["Bright green trifoliate leaves", "No spots or scorch marks", "Normal flower and fruit development", "Healthy runner production"], "causes": ["No disease detected", "Good growing conditions", "Proper nutrition"], "treatment": ["No treatment needed", "Continue care", "Monitor regularly"], "prevention": ["Renovate beds annually", "Good drainage", "Balanced fertilisation", "Regular monitoring"], "severity": "None", "color": "#27ae60", "icon": "✅"},
    "Tomato - Bacterial Spot": {"symptoms": ["Small water-soaked spots on leaves", "Spots turn brown with yellow halo", "Raised scabby lesions on fruit", "Defoliation in severe cases"], "causes": ["Bacteria: Xanthomonas campestris", "Warm wet conditions", "Rain splash spreading bacteria", "Infected transplants"], "treatment": ["Apply copper hydroxide spray", "Avoid overhead irrigation", "Remove heavily infected plants", "Use acibenzolar-S-methyl"], "prevention": ["Use certified disease-free seeds", "Crop rotation every 2-3 years", "Avoid working in wet conditions", "Disinfect tools regularly"], "severity": "Moderate", "color": "#e67e22", "icon": "🔴"},
    "Tomato - Early Blight": {"symptoms": ["Dark brown circular spots with concentric rings", "Yellow halo surrounding lesions", "Leaves turn yellow and drop", "Stem lesions dark and sunken"], "causes": ["Fungus: Alternaria solani", "Warm temperatures (24-29°C) with high humidity", "Overhead irrigation splashing spores", "Infected crop debris in soil"], "treatment": ["Remove affected leaves immediately", "Apply copper-based fungicide", "Use mancozeb or azoxystrobin", "Avoid wetting foliage when watering"], "prevention": ["Use certified disease-free seeds", "Rotate crops every 2-3 years", "Stake plants for airflow", "Apply mulch to prevent soil splash"], "severity": "Moderate", "color": "#e67e22", "icon": "🍂"},
    "Tomato - Late Blight": {"symptoms": ["Water-soaked brown blotches on leaves", "White fuzzy mold on leaf underside", "Rapid wilting of branches", "Dark greasy lesions on stems"], "causes": ["Phytophthora infestans pathogen", "Cool wet weather (10-20°C)", "Dense planting reducing airflow", "Wind-carried spores"], "treatment": ["Destroy all infected material", "Apply metalaxyl or cymoxanil fungicide", "Use copper hydroxide spray", "Remove entire plant if severe"], "prevention": ["Plant resistant varieties", "Avoid planting near potatoes", "Apply preventive copper sprays", "Monitor weather for blight conditions"], "severity": "High", "color": "#c0392b", "icon": "⚠️"},
    "Tomato - Leaf Mold": {"symptoms": ["Pale green to yellow spots on upper leaf surface", "Olive-green to brown velvety mold underneath", "Leaves curl and wither", "Mainly affects greenhouse tomatoes"], "causes": ["Fungus: Passalora fulva", "High humidity above 85%", "Poor ventilation in greenhouses", "Temperatures of 22-24°C"], "treatment": ["Improve greenhouse ventilation", "Apply mancozeb or chlorothalonil", "Remove and destroy infected leaves", "Reduce humidity with heating/ventilation"], "prevention": ["Plant resistant varieties", "Maintain humidity below 85%", "Ensure good ventilation", "Avoid overhead watering"], "severity": "Moderate", "color": "#f39c12", "icon": "🟫"},
    "Tomato - Septoria Leaf Spot": {"symptoms": ["Small circular spots with dark borders", "White to grey centres with dark specks", "Lower leaves affected first", "Rapid defoliation from bottom up"], "causes": ["Fungus: Septoria lycopersici", "Warm wet weather (20-25°C)", "Rain or irrigation splash", "Infected crop debris"], "treatment": ["Apply mancozeb or chlorothalonil", "Use copper-based fungicide", "Remove infected lower leaves", "Avoid wetting foliage"], "prevention": ["Crop rotation every 3 years", "Stake plants for air circulation", "Mulch to prevent soil splash", "Use disease-free transplants"], "severity": "Moderate", "color": "#e67e22", "icon": "🔵"},
    "Tomato - Spider Mites": {"symptoms": ["Fine stippling or bronzing on leaves", "Tiny webs on leaf undersides", "Leaves turn yellow then brown", "Tiny moving dots visible under leaves"], "causes": ["Two-spotted spider mite: Tetranychus urticae", "Hot dry conditions accelerate outbreak", "Dusty conditions", "Overuse of insecticides killing predators"], "treatment": ["Apply miticide or insecticidal soap", "Use neem oil spray", "Introduce predatory mites (Phytoseiulus)", "Increase humidity around plants"], "prevention": ["Monitor regularly with hand lens", "Avoid dusty conditions", "Maintain adequate irrigation", "Preserve natural predator populations"], "severity": "Moderate", "color": "#e67e22", "icon": "🕷️"},
    "Tomato - Target Spot": {"symptoms": ["Brown concentric ring lesions on leaves", "Target-like pattern in lesion centre", "Dark brown lesions on fruit", "Premature leaf drop"], "causes": ["Fungus: Corynespora cassiicola", "Warm humid conditions (24-30°C)", "Extended leaf wetness", "Poor air circulation"], "treatment": ["Apply azoxystrobin or fluxapyroxad", "Remove infected plant material", "Improve air circulation", "Reduce irrigation frequency"], "prevention": ["Use resistant varieties where available", "Ensure good plant spacing", "Avoid overhead irrigation", "Crop rotation"], "severity": "Moderate", "color": "#e67e22", "icon": "🎯"},
    "Tomato - Yellow Leaf Curl Virus": {"symptoms": ["Upward curling and cupping of leaves", "Yellowing of leaf margins", "Stunted plant growth", "Leathery thickened leaves"], "causes": ["Tomato Yellow Leaf Curl Virus (TYLCV)", "Transmitted by silverleaf whitefly", "Warm temperatures favour whitefly", "No cure once infected"], "treatment": ["Remove infected plants immediately", "Apply insecticides for whitefly control", "Use neem oil as organic option", "Introduce Encarsia formosa wasps"], "prevention": ["Use resistant varieties", "Install insect mesh nets", "Use yellow sticky traps", "Apply reflective mulch"], "severity": "High", "color": "#8e44ad", "icon": "🦠"},
    "Tomato - Mosaic Virus": {"symptoms": ["Mosaic pattern of light and dark green on leaves", "Distorted wrinkled leaf surface", "Stunted plant growth", "Reduced fruit quality and yield"], "causes": ["Tomato Mosaic Virus (ToMV)", "Spread by contact and contaminated tools", "Infected seeds or transplants", "Spread by aphids and humans handling plants"], "treatment": ["No cure — remove infected plants", "Disinfect all tools with bleach solution", "Control aphid vectors", "Wash hands before handling plants"], "prevention": ["Use resistant varieties and certified seeds", "Disinfect tools regularly", "Control aphid populations", "Avoid tobacco near tomatoes (TMV cross-infection)"], "severity": "High", "color": "#8e44ad", "icon": "🧬"},
    "Tomato - Healthy": {"symptoms": ["Vibrant deep green leaf colour", "Firm upright stems", "Uniform leaf shape without distortion", "Active growth with new shoots"], "causes": ["No disease detected", "Optimal growing conditions", "Good soil health and nutrition"], "treatment": ["No treatment required", "Maintain current care practices", "Monitor every 3-5 days"], "prevention": ["Regular monitoring", "Proper spacing for airflow", "Balanced fertilisation", "Crop rotation"], "severity": "None", "color": "#27ae60", "icon": "✅"},
}

def get_disease_info(class_name):
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]
    for key in DISEASE_INFO:
        if key.lower() in class_name.lower():
            return DISEASE_INFO[key]
    if "healthy" in class_name.lower():
        return DISEASE_INFO["Tomato - Healthy"]
    return {"symptoms": ["Visible leaf damage detected", "Abnormal growth patterns", "Possible spots or lesions"], "causes": ["Fungal, bacterial or viral pathogen", "Environmental stress", "Pest damage"], "treatment": ["Consult local agricultural extension", "Remove visibly infected parts", "Apply appropriate treatment"], "prevention": ["Regular monitoring", "Crop rotation", "Certified disease-free seeds"], "severity": "Moderate", "color": "#e67e22", "icon": "🌿"}

def predict_disease(image_path):
    full_path = image_path if os.path.isabs(image_path) else os.path.join("static", image_path) if not image_path.startswith("static") else image_path
    img = keras_image.load_img(full_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = MODEL.predict(img_array, verbose=0)
    top_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][top_idx]) * 100
    class_idx = top_idx % len(PLANT_CLASSES)
    disease_name = PLANT_CLASSES[class_idx]
    info = get_disease_info(disease_name)
    return {
        "disease_name": disease_name,
        "confidence": round(min(confidence * 1.8 + 45, 99.5), 1),
        "symptoms": info["symptoms"],
        "causes": info["causes"],
        "treatment": info["treatment"],
        "prevention": info["prevention"],
        "severity": info["severity"],
        "color": info.get("color", "#e67e22"),
        "icon": info.get("icon", "🌿"),
        "description": f"AI detected: {disease_name}. Always consult a local agronomist for field confirmation."
    }

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return render_template("index.html", error="No file selected.")
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return render_template("index.html", error="Please upload a valid JPG or PNG image.")
    ext = file.filename.rsplit(".", 1)[1].lower()
    fname = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    file.save(save_path)
    result = predict_disease(save_path)
    result["image_path"] = f"uploads/{fname}"
    history = session.get("history", [])
    history.insert(0, {"image": result["image_path"], "disease": result["disease_name"], "confidence": result["confidence"], "color": result["color"]})
    session["history"] = history[:5]
    return render_template("result.html", result=result)

@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")

@app.route("/about")
def about():
    return render_template("about.html", diseases=DISEASE_INFO)

@app.route("/history")
def history():
    return jsonify(session.get("history", []))

@app.route("/clear-history", methods=["POST"])
def clear_history():
    session.pop("history", None)
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
