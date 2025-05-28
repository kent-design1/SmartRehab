# SmartRehab UI

This is the front-end and API implementation for the SmartRehab system — a predictive and explainable rehabilitation decision support platform. It enables users to input patient features, receive SCIM score predictions, and explore explainability via SHAP.

---

## 📁 Project Structure

### Backend
- **`app.py`**: FastAPI backend serving prediction and SHAP explanations.

### Frontend (React / Next.js + TailwindCSS)
- **`About.tsx`**: Static page providing project description and context.
- **`FeatureForm.tsx`**: Main form where user inputs rehab features.
- **`PredictionResult.tsx`**: Displays predicted SCIM output and recommendations.
- **`Tabs.tsx`**: Handles tab navigation for therapy, costs, and results.
- **`Nav.tsx`**: Top navigation bar.
- **`Footer.tsx`**: App footer.
- **`Hero.tsx`**: Homepage hero section.
- **`page.tsx`**: Main routing entry point and container for layout.
- **`route.ts`**: API routing logic for backend interaction.

---

## 🚀 Usage

### 1. Backend (FastAPI)
```bash
cd backend
uvicorn app:app --reload
```

### 2. Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
```

---

## 📦 API Endpoint Example

### `POST /predict/{{week}}`
**Input JSON:**
```json
{{
  "features": {{
    "Age": 62,
    "SessionsPerWeek": 3,
    "HealthCondition": "Stroke Recovery",
    "Total_SCIM_0": 25,
    "TherapyPlan": ["Hydrotherapy", "Strength & FES Training"]
  }}
}}
```

**Output JSON:**
```json
{{
  "week": 12,
  "prediction": 48.5,
  "cost": {{
    "baseline_cost": 11520.0,
    "overuse_cost": 960.0,
    "total_cost": 12480.0,
    "efficiency": 0.00388
  }},
  "static_recommendations": [...],
  "shap_recommendations": [...]
}}
```

---

## 🔐 Privacy
Patient data is synthetic and used only for academic prototyping. SHAP values are computed in real-time per patient input.

---

## 📄 License
MIT License

---

## 📬 Contact
Created by Israel Kenneth Asamoah Bamfo  
University of Bern, 2025
