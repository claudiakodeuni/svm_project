import os
import re
import pandas as pd

# Paths to raw data directories
lapaz_path="data/raw/la_paz"
quito_path="data/raw/quito"

# List of interjections to remove
INTERJECTIONS = {
    "aah", "aam", "abb", "ah", "ahh", "aj", "ajajá", "ajá", "am", "amm", "ay",
    "bah", "buah", "eeeh", "eeh", "eem", "eh", "ehm", "ehhh", "ejem", "emm",
    "epa", "ey", "hala", "hum", "huy", "jajaja", "mhm", "mmh", "mmm", "mmmm",
    "oh","olé", "pst", "pche", "pchs", "puaf", "puaff", "puaj", "puf", "puff",
    "uf", "uff", "uh", "uhm", "uhu", "uhum", "ujum", "um", "umm", "wow"
}

# Blacklist with Entities
BLACKLIST = {
    "Achacachi", "Achachicala", "Achocalla", "Achumani",
    "Alausí", "Alto Obra", "Alto Obrajes", "Alto Pacasa", "Alto Següencoma",
    "Ambato", "Atahualpa Mena Chillogallo", "Avaroa",
    "Banco Ecuatoriano de la Vivienda", "Baños de Agua Santa",
    "Barrientos", "Bellavista", "Beni", "Bush",
    "Calacoto", "Camacho", "Carcelén",
    "Chacaltaya", "Chacaltaya de la Zona Norte", "Challapata",
    "Charaña", "Chasquipampa", "Chillogallo", "Chimoré",
    "Ciudad del Niño", "Club Bolívar", "Coca", "Cobija",
    "Cochabamba", "Cochabambo", "Copacabana", "Copacabano",
    "Coroico", "Cuenca", "Cumbayá", "Cyrano",
    "Digitech", "Domsat",
    "Ecovía", "El Alto", "Embajada de Bolivia", "Esmeraldas",
    "Guápulo", "Guaranda", "Guayaquil",
    "Hampaturi", "Huayna Potosí",
    "Ibarra", "IESS", "IETEL",
    "Katari", "Kañuma", "Kenko",
    "La Ceja", "La Paz", "La Paz Líder", "La Paz Oruro",
    "La Paz Oruro Potosí Santa Cruz Beni",
    "Lago Agrio", "Latacunga", "Loja", "Loja Cuenca", "Loja Quito",
    "Luribay",
    "Machachi", "Machala", "Mallasa", "Metro de Quito",
    "Milluni", "Mindo", "Montañita", "Monterrey",
    "Munaypata", "Murillo",
    "Obrajes", "Oruro", "Otavalo",
    "Pacari", "Pacasa", "Palca", "Pallatanga", "Pampahasi",
    "Pando", "Pichincho", "Potosí", "Potosí Sucre",
    "Princes", "Provincia Murillo", "Puerto Quito", "Puembo", "Puyo",
    "Quevedo", "Quito",
    "René", "Riberalta", "Rumiñahui", "Rurrenabaque",
    "Salar de Uyuni", "San Agustín", "San Pedro", "Santa Cruz",
    "Shell Mera", "Shushufindi", "Solanda", "Sorata",
    "Sur Yungas", "Surani",
    "Tabacundo", "Tambo Quema", "Tarija", "Tena",
    "Tiwanaku", "Toacazo", "Todos Santos", "Tumbaco",
    "Tumusla", "Tupiza",
    "Universidad Particular de Loja", "Uyuni",
    "Villazón",
    "Yacuiba", "Yunga", "Yungas"
}

# Build a regex that matches any BLACKLIST entity (case-insensitive)
BLACKLIST_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(e) for e in sorted(BLACKLIST, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE
)

def clean_line(line, remove_blacklist=True):
    """
    Cleans a line by:
    - Removing transcription markup (<…>)
    - Removing 'I:' prefix
    - Replacing blacklist entities with 'entity'
    - Removing interjections
    - Lowercasing
    - Removing punctuation, numbers, special characters
    - Stripping whitespace
    """

    # Remove anything between < and >
    line = re.sub(r"<[^>]+>", "", line)

    # Remove leading "I:"
    line = line.lstrip("I:").strip()

    # Replace blacklist entities BEFORE lowercasing
    if remove_blacklist:
        line = BLACKLIST_PATTERN.sub("entity", line)

    # Remove interjections
    words = line.split()
    words = [w for w in words if w.lower() not in INTERJECTIONS]
    line = " ".join(words)

    # Lowercase
    line = line.lower()

    # Remove punctuation, numbers, special characters
    line = re.sub(r"[^a-záéíóúüñ\s]", "", line)

    # Strip extra spaces
    line = re.sub(r"\s+", " ", line).strip()

    return line

def label_interviewee_speech(lapaz_path="data/raw/la_paz", quito_path="data/raw/quito", remove_blacklist=True):
    labeled_data = []

    def process_folder(folder_path, label):
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("I:"):
                            line = clean_line(line, remove_blacklist=remove_blacklist)
                            if line:
                                labeled_data.append({"line": line, "label": label})

    process_folder(lapaz_path, "lapaz")
    process_folder(quito_path, "quito")

    return labeled_data

# Process data and save to CSV
data = label_interviewee_speech()
df = pd.DataFrame(data)

out_dir = os.path.join("data", "clean")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "cleaned_data.csv")

df.to_csv(out_path, index=False, encoding="utf-8")

print(f"CSV saved as '{out_path}'")
