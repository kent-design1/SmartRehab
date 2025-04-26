import { useState } from 'react';
import { getPrediction } from '@/lib/api';

interface FeatureFormProps {
    week: number;
    onResult: (value: number) => void;
}

// SCIM fields per week
const requiredScim: Record<number, string[]> = {
    6: ['Total_SCIM_0'],
    12: ['Total_SCIM_0'],
    18: ['Total_SCIM_0'],
    24: ['Total_SCIM_0'],
};
const optionalScim: Record<number, string[]> = {
    6: [],
    12: ['Total_SCIM_6'],
    18: ['Total_SCIM_6', 'Total_SCIM_12'],
    24: ['Total_SCIM_6', 'Total_SCIM_12', 'Total_SCIM_18'],
};

// Baseline features common to all
const baselineFields: { key: string; label: string; type: 'number' | 'text' }[] = [
    { key: 'Age', label: 'Age', type: 'number' },
    { key: 'Gender', label: 'Gender', type: 'text' },
    { key: 'Ethnicity', label: 'Ethnicity', type: 'text' },
    { key: 'Education', label: 'Education', type: 'text' },
    { key: 'Insurance', label: 'Insurance', type: 'text' },
    { key: 'Region', label: 'Region', type: 'text' },
    { key: 'Height_cm', label: 'Height (cm)', type: 'number' },
    { key: 'Weight_kg', label: 'Weight (kg)', type: 'number' },
    { key: 'BMI', label: 'BMI', type: 'number' },
    { key: 'CRP_mg_L', label: 'CRP (mg/L)', type: 'number' },
    { key: 'Hemoglobin_g_dL', label: 'Hemoglobin (g/dL)', type: 'number' },
    { key: 'Glucose_mg_dL', label: 'Glucose (mg/dL)', type: 'number' },
    { key: 'PHQ9', label: 'PHQ-9 Score', type: 'number' },
    { key: 'GAD7', label: 'GAD-7 Score', type: 'number' },
    { key: 'Sessions per Week', label: 'Sessions per Week', type: 'number' },
    { key: 'Engagement Score', label: 'Engagement Score', type: 'number' },
    { key: 'Mental Health Scale', label: 'Mental Health Scale', type: 'number' },
    { key: 'Pain Level', label: 'Pain Level', type: 'number' },
    { key: 'Fatigue Level', label: 'Fatigue Level', type: 'number' },
    { key: 'Muscle Strength Upper', label: 'Upper Muscle Strength', type: 'number' },
    { key: 'Muscle Strength Lower', label: 'Lower Muscle Strength', type: 'number' },
    { key: 'Balance Test Score', label: 'Balance Test', type: 'number' },
    { key: 'Mobility Test Score', label: 'Mobility Test', type: 'number' },
    { key: 'Total Estimated Cost', label: 'Estimated Cost', type: 'number' },
];

export default function FeatureForm({ week, onResult }: FeatureFormProps) {
    // Determine SCIM fields
    const req = requiredScim[week] || [];
    const opt = optionalScim[week] || [];
    const scimFields = [...req, ...opt];

    // Initialize state for all fields
    const initialValues: Record<string, string> = {};
    baselineFields.forEach(f => (initialValues[f.key] = ''));
    scimFields.forEach(key => (initialValues[key] = ''));

    const [values, setValues] = useState(initialValues);
    const [loading, setLoading] = useState(false);

    const handleChange = (key: string, val: string) => {
        setValues(prev => ({ ...prev, [key]: val }));
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        // build payload with numbers and strings
        const payload: Record<string, any> = {};
        baselineFields.forEach(f => {
            if (values[f.key]) {
                payload[f.key] = f.type === 'number' ? parseFloat(values[f.key]) : values[f.key];
            }
        });
        scimFields.forEach(key => {
            if (values[key]) payload[key] = parseFloat(values[key]);
        });

        try {
            const pred = await getPrediction(week, payload);
            onResult(pred);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    // disable if missing req SCIM
    const invalid = req.some(key => !values[key]);

    return (
        <form onSubmit={handleSubmit} className="space-y-6">
            {/* Baseline fields */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {baselineFields.map(f => (
                    <div key={f.key}>
                        <label className="block text-sm font-medium text-gray-700">{f.label}</label>
                        <input
                            type={f.type}
                            value={values[f.key]}
                            onChange={e => handleChange(f.key, e.target.value)}
                            className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                        />
                    </div>
                ))}
            </div>

            {/* SCIM fields */}
            <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                {scimFields.map(key => (
                    <div key={key}>
                        <label className="block text-sm font-medium text-gray-700">
                            {key}{req.includes(key) && ' *'}
                        </label>
                        <input
                            type="number"
                            step="0.1"
                            value={values[key]}
                            onChange={e => handleChange(key, e.target.value)}
                            className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                        />
                    </div>
                ))}
            </div>

            <button
                type="submit"
                disabled={invalid || loading}
                className="mt-4 w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50"
            >
                {loading ? 'Predicting...' : 'Predict for Week ' + week}
            </button>
        </form>
    );
}
