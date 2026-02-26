# ⚡ ENIGMA 2.0 - Wattwatchers

A smart AI-powered pipeline for lunar mapping, landslide detection, boulder detection, age classification, and real-time visualization — built for ENIGMA 2.0 Hackathon.

---

## 🌕 Core Notebook

### [`ENIGMA_2_0_LunarMapper.ipynb`](./notebooks/ENIGMA_2_0_LunarMapper.ipynb)

> **Start here.** This is the primary notebook for the ENIGMA 2.0 project — it brings together the full lunar mapping pipeline in an interactive, end-to-end Jupyter environment.

- 🌍 End-to-end lunar surface analysis workflow
- 🪨 Integrated boulder & landslide detection
- 📊 Inline visualizations and result exports
- 🔬 Ideal for exploration, experimentation, and demo

**To run the notebook:**
```bash
jupyter notebook notebooks/ENIGMA_2_0_LunarMapper.ipynb
```

---

## 🚀 Features

- 🌕 **Lunar Mapper Notebook** — Full interactive pipeline in one notebook *(primary entry point)*
- 🪨 **Boulder Detection** — Detects boulders in terrain using computer vision
- 🌊 **Landslide Detection** — Identifies landslide-prone areas from input data
- 🧑 **Age Classifier** — Classifies age from visual input
- 📊 **Interactive Dashboard** — Real-time visual dashboard (`dashboard_v2.html`)
- 🔁 **Pipeline Architecture** — Modular end-to-end processing pipeline
- 📤 **Exporter** — Export results in structured formats

---

## 🗂️ Project Structure

```
ENIGMA_2.0_Wattwatchers/
│
├── notebooks/
│   └── ENIGMA_2_0_LunarMapper.ipynb  # ⭐ Main project notebook
│
├── src/
│   ├── pipeline.py           # Main pipeline orchestrator
│   ├── landslide_detector.py # Landslide detection module
│   ├── boulder_detector.py   # Boulder detection module
│   ├── age_classifier.py     # Age classification module
│   ├── visualizer.py         # Visualization utilities
│   ├── preprocess.py         # Data preprocessing
│   └── exporter.py           # Result exporter
│
├── config/
│   └── config.yaml           # Configuration file
│
├── tests/
│   └── test_pipeline.py      # Unit tests
│
├── dashboard_v2.html         # Interactive web dashboard
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/sohammxx/ENIGMA_2.0_Wattwatchers.git
cd ENIGMA_2.0_Wattwatchers
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

**Run the main notebook *(recommended)*:**
```bash
jupyter notebook notebooks/ENIGMA_2_0_LunarMapper.ipynb
```

Run the pipeline script:
```bash
python src/pipeline.py
```

Open the dashboard:
- Simply open `dashboard_v2.html` in your browser

Run tests:
```bash
python -m pytest tests/
```

---

## ⚙️ Configuration

Edit `config/config.yaml` to adjust pipeline settings like input paths, model parameters, and output formats.

---

## 👥 Team

Built with ❤️ by Team Wattwatchers for ENIGMA 2.0 Hackathon.

---

## 📄 License

This project is for hackathon purposes only.
