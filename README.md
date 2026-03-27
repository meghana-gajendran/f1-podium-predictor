#  F1 Podium Predictor

> Formula 1 podium predictions using live qualifying data and machine learning.

---

##  Live Demo
https://f1-podium-predictor-drxupsyvygdansfhyy95kj.streamlit.app/

---

##  Features

-  **Dark F1-themed UI** — red accents, race atmosphere, Barlow Condensed font
-  **Live qualifying grid** — auto-fetched from OpenF1 API
-  **Podium cards** — gold / silver / bronze styled results
-  **Two Plotly charts** — probability bars + score breakdown
-  **Full ranked table** — color-gradient driver standings
-  **Cache controls** — one-click data refresh

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Data | FastF1, OpenF1 API |
| ML | scikit-learn |
| Charts | Plotly |
| Language | Python 3.9+ |

---

## Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/your-username/f1-podium-predictor.git
cd f1-podium-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run f1_app.py
```

Opens at `http://localhost:8501` 

---


##  Project Structure

```
f1-podium-predictor/
├── f1_app.py          # Main Streamlit application
├── requirements.txt   # Python dependencies
└── README.md          # You are here
```

---

##  Requirements

```
streamlit
plotly
fastf1
scikit-learn
requests
pandas
```

---

## License

[MIT](https://choosealicense.com/licenses/mit/)

---

## Acknowledgements

- [FastF1](https://github.com/theOehrly/Fast-F1) — F1 telemetry and timing data
- [OpenF1 API](https://openf1.org) — live F1 data
- [Streamlit](https://streamlit.io) — app framework
