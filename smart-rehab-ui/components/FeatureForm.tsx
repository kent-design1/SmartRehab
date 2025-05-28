//predict/FeatureForm
'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { getPrediction, PredictionResponse } from '@/lib/api';

interface FeatureFormProps {
    week: number;
}

type SCIMMap = Record<number, string[]>;
const requiredScim: SCIMMap = {
    6: ['Total_SCIM_0'],
    12: ['Total_SCIM_0'],
    18: ['Total_SCIM_0'],
    24: ['Total_SCIM_0']
};
const optionalScim: SCIMMap = {
    6: [],
    12: ['Total_SCIM_6'],
    18: ['Total_SCIM_6', 'Total_SCIM_12'],
    24: ['Total_SCIM_6', 'Total_SCIM_12', 'Total_SCIM_18']
};

const healthConditions = [
    'Stroke Recovery',
    'Orthopedic Injury',
    'Chronic Pain',
    'Diabetes',
    'Hypertension'
] as const;
type Cond = typeof healthConditions[number];

const therapyPlanOptions: Record<Cond, string[]> = {
    'Stroke Recovery': [
        'Respiratory Management',
        'Spasticity Management',
        'Mobility & Upper Limb Training',
        'Strength & FES Training',
        'Comprehensive SCI Rehab',
        'Robotic Gait Training',
        'Virtual Reality Therapy',
        'Aquatic Therapy',
        'Constraint-Induced Movement Therapy',
        'Mirror Therapy',
    ],
    'Orthopedic Injury': [
        'Physiotherapy',
        'Occupational Therapy',
        'Manual Therapy',
        'Hydrotherapy',
        'Proprioceptive Neuromuscular Facilitation',
        'Functional Strength Training',
        'Balance & Proprioception Drills',
    ],
    'Chronic Pain': [
        'Physiotherapy',
        'Medication',
        'Lifestyle Changes',
        'Cognitive Behavioral Therapy',
        'Mindfulness-Based Stress Reduction',
        'Pain Neuroscience Education',
        'Yoga/Tai-Chi',
        'Graded Activity/Exposure',
    ],
    'Diabetes': [
        'Medication',
        'Lifestyle Changes',
        'Nutritional Counseling & Meal Planning',
        'Aerobic Exercise Program',
        'Resistance Training',
        'Foot Care & Offloading Education',
        'Blood Glucose–Guided Activity',
    ],
    'Hypertension': [
        'Medication',
        'Lifestyle Changes',
        'Structured Aerobic Training',
        'Resistance Exercise',
        'Stress Management',
        'Dietary Sodium Reduction',
        'Tele-rehab / Remote Monitoring',
    ],
};

interface FieldMeta {
    key: string;
    label: string;
    type: 'number' | 'select' | 'multiselect';
    options?: string[];
    placeholder?: string;
}

const baselineFields: FieldMeta[] = [
    { key: 'Age', label: 'Age', type: 'number', placeholder: '65' },
    { key: 'Gender', label: 'Gender', type: 'select', options: ['Male', 'Female', 'Other'] },
    { key: 'Ethnicity', label: 'Ethnicity', type: 'select', options: ['Swiss', 'German', 'French', 'Italian', 'Other'] },
    {
        key: 'Education',
        label: 'Education',
        type: 'select',
        options: ['Compulsory', 'Apprenticeship', 'Vocational', 'Bachelor', 'Master/Doctorate', 'Other']
    },
    {
        key: 'Insurance',
        label: 'Insurance',
        type: 'select',
        options: ['Basic Mandatory', 'Supplementary Private', 'Employer-Sponsored', 'Uninsured', 'Other']
    },
    { key: 'Region', label: 'Region', type: 'select', options: ['Zurich', 'Bern', 'Geneva', 'Vaud', 'Other'] },

    { key: 'HealthCondition', label: 'Health Condition', type: 'select', options: [...healthConditions] },
    { key: 'TherapyPlan', label: 'Current Therapy Plan(s)', type: 'multiselect', options: [] },

    { key: 'SessionsPerWeek', label: 'Sessions/Week', type: 'select', options: Array.from({ length: 8 }, (_, i) => i.toString()) },
    { key: 'CharlsonIndex', label: 'Charlson Index', type: 'number', placeholder: '1' },
    { key: 'Albumin_g_dL', label: 'Albumin (g/dL)', type: 'number', placeholder: '4.0' },
    { key: 'TUG_sec', label: 'TUG (sec)', type: 'number', placeholder: '12.5' },
    { key: 'CRP_mg_L', label: 'CRP (mg/L)', type: 'number', placeholder: '5.2' },
    { key: 'Hemoglobin_g_dL', label: 'Hemoglobin (g/dL)', type: 'number', placeholder: '13.5' },
    { key: 'Glucose_mg_dL', label: 'Glucose (mg/dL)', type: 'number', placeholder: '100' },
    { key: 'PHQ9', label: 'PHQ-9 Score', type: 'number', placeholder: '0–27' },
    { key: 'GAD7', label: 'GAD-7 Score', type: 'number', placeholder: '0–21' },
    { key: 'PainLevel', label: 'Pain Level (1–10)', type: 'number', placeholder: '1–10' },
    { key: 'FatigueLevel', label: 'Fatigue Level (1–10)', type: 'number', placeholder: '1–10' },
    { key: 'MuscleStrengthUpper', label: 'Upper Strength (1–10)', type: 'number', placeholder: '1–10' },
    { key: 'MuscleStrengthLower', label: 'Lower Strength (1–10)', type: 'number', placeholder: '1–10' },
    { key: 'BalanceTestScore', label: 'Balance Test (1–100)', type: 'number', placeholder: '1–100' },
    { key: 'MobilityTestScore', label: 'Mobility Test (1–100)', type: 'number', placeholder: '1–100' }
];

export default function FeatureForm({ week }: FeatureFormProps) {
    const router = useRouter();
    const req = requiredScim[week] || [];
    const opt = optionalScim[week] || [];
    const scimFields = [...req, ...opt];

    // initialize form state
    const initial: Record<string, any> = {};
    baselineFields.forEach(f => (initial[f.key] = f.type === 'multiselect' ? [] : ''));
    scimFields.forEach(k => (initial[k] = ''));

    const [values, setValues] = useState(initial);
    const [loading, setLoading] = useState(false);

    // update therapy options when condition changes
    useEffect(() => {
        const cond = values.HealthCondition as Cond;
        const opts = cond ? therapyPlanOptions[cond] : [];
        baselineFields.find(f => f.key === 'TherapyPlan')!.options = opts;
        setValues(v => ({ ...v, TherapyPlan: [] }));
    }, [values.HealthCondition]);

    const togglePlan = (plan: string) =>
        setValues(v => ({
            ...v,
            TherapyPlan: v.TherapyPlan.includes(plan)
                ? v.TherapyPlan.filter((x: string) => x !== plan)
                : [...v.TherapyPlan, plan]
        }));

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);

        // build payload
        const payload: Record<string, any> = {};
        baselineFields.forEach(f => {
            const v = values[f.key];
            if (f.type === 'number' && v !== '') payload[f.key] = +v;
            else if (f.type === 'select' && v !== '') payload[f.key] =
                f.key === 'SessionsPerWeek' ? +v : v;
            else if (f.type === 'multiselect' && v.length) payload[f.key] = v;
        });
        scimFields.forEach(k => {
            const v = values[k];
            if (v !== '') payload[k] = +v;
        });

        try {
            const resp: PredictionResponse = await getPrediction(week, payload);

            // serialize everything into the query‑string
            const qs = new URLSearchParams();
            qs.set('week',       resp.week.toString());
            qs.set('prediction', resp.prediction.toString());
            qs.set('cost',        encodeURIComponent(JSON.stringify(resp.cost)));
            qs.set('static_recs', encodeURIComponent(JSON.stringify(resp.static_recommendations)));
            qs.set('shap_recs',   encodeURIComponent(JSON.stringify(resp.shap_recommendations)));

            router.push(`/predict/results?${qs.toString()}`);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const invalid = req.some(k => !values[k]);


    return (
        <form onSubmit={handleSubmit} className="space-y-10">
            <h2 className="text-2xl font-bold text-gray-800">🏥 Patient & Therapy Details</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {baselineFields.map(f => (
                    <div key={f.key} className="flex flex-col">
                        <label className="mb-1 font-medium text-gray-700">{f.label}</label>
                        {f.type === 'multiselect' && f.options ? (
                            <div className="border rounded-lg p-2 grid grid-cols-2 gap-2 max-h-40 overflow-auto">
                                {f.options.map(opt => (
                                    <label key={opt} className="flex items-center space-x-2">
                                        <input
                                            type="checkbox"
                                            checked={values.TherapyPlan.includes(opt)}
                                            onChange={() => togglePlan(opt)}
                                            className="h-4 w-4 text-indigo-600"
                                        />
                                        <span className="text-gray-700 text-sm">{opt}</span>
                                    </label>
                                ))}
                            </div>
                        ) : f.type === 'select' && f.options ? (
                            <select
                                value={values[f.key]}
                                onChange={e => setValues(v => ({ ...v, [f.key]: e.target.value }))}
                                className="border rounded-lg px-3 py-2 focus:ring-indigo-500 focus:border-indigo-500"
                            >
                                <option value="" disabled>
                                    Select {f.label}
                                </option>
                                {f.options.map(o => (
                                    <option key={o} value={o}>
                                        {o}
                                    </option>
                                ))}
                            </select>
                        ) : (
                            <input
                                type="number"
                                placeholder={f.placeholder}
                                value={values[f.key]}
                                onChange={e => setValues(v => ({ ...v, [f.key]: e.target.value }))}
                                className="border rounded-lg px-3 py-2 focus:ring-indigo-500 focus:border-indigo-500"
                            />
                        )}
                    </div>
                ))}
            </div>

            <h2 className="text-2xl font-bold text-gray-800">📊 Baseline SCIM Score</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {scimFields.map(k => (
                    <div key={k} className="flex flex-col">
                        <label className="mb-1 font-medium text-gray-700">
                            {k}
                            {requiredScim[week].includes(k) && <span className="text-red-500">*</span>}
                        </label>
                        <input
                            type="number"
                            step="0.1"
                            placeholder={requiredScim[week].includes(k) ? 'Required' : 'Optional'}
                            value={values[k]}
                            onChange={e => setValues(v => ({ ...v, [k]: e.target.value }))}
                            className="border rounded-lg px-3 py-2 focus:ring-indigo-500 focus:border-indigo-500"
                        />
                    </div>
                ))}
            </div>

            <button
                type="submit"
                disabled={invalid || loading}
                className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-3 text-lg font-semibold rounded-lg shadow transition disabled:opacity-50"
            >
                {loading ? 'Predicting…' : `Predict Week ${week}`}
            </button>
        </form>
    );
}