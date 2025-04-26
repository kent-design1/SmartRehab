import { useState } from 'react';
import { getPrediction } from '@/lib/api';

interface FeatureFormProps {
    week: number;
    onResult: (value: number) => void;
}

// SCIM fields per week
type SCIMMap = Record<number, string[]>;
const requiredScim: SCIMMap = { 6: ['Total_SCIM_0'], 12: ['Total_SCIM_0'], 18: ['Total_SCIM_0'], 24: ['Total_SCIM_0'] };
const optionalScim: SCIMMap = { 6: [], 12: ['Total_SCIM_6'], 18: ['Total_SCIM_6','Total_SCIM_12'], 24: ['Total_SCIM_6','Total_SCIM_12','Total_SCIM_18'] };

// Enhanced baseline with select options and placeholders
interface FieldMeta {
    key: string;
    label: string;
    type: 'number' | 'text' | 'select';
    placeholder?: string;
    options?: string[];
}
const baselineFields: FieldMeta[] = [
    { key: 'Age', label: 'Age', type: 'number', placeholder: 'e.g. 65' },
    { key: 'Gender', label: 'Gender', type: 'select', options: ['Male','Female','Other'] },
    { key: 'Ethnicity', label: 'Ethnicity', type: 'select', options: ['Swiss','German','French','Italian','Other'] },
    { key: 'Education', label: 'Education', type: 'select', options: ['Compulsory','Apprenticeship','Vocational','Bachelor','Master/Doctorate','Other'] },
    { key: 'Insurance', label: 'Insurance', type: 'select', options: ['Basic Mandatory','Supplementary Private','Employer-Sponsored','Uninsured','Other'] },
    { key: 'Region', label: 'Region', type: 'select', options: ['Zurich','Bern','Geneva','Vaud','Other'] },
    { key: 'Height_cm', label: 'Height (cm)', type: 'number', placeholder: 'e.g. 170' },
    { key: 'Weight_kg', label: 'Weight (kg)', type: 'number', placeholder: 'e.g. 70' },
    { key: 'BMI', label: 'BMI', type: 'number', placeholder: 'e.g. 24.5' },
    { key: 'CRP_mg_L', label: 'CRP (mg/L)', type: 'number', placeholder: 'e.g. 5.2' },
    { key: 'Hemoglobin_g_dL', label: 'Hemoglobin (g/dL)', type: 'number', placeholder: 'e.g. 13.5' },
    { key: 'Glucose_mg_dL', label: 'Glucose (mg/dL)', type: 'number', placeholder: 'e.g. 100' },
    { key: 'PHQ9', label: 'PHQ-9 Score', type: 'number', placeholder: '0-27' },
    { key: 'GAD7', label: 'GAD-7 Score', type: 'number', placeholder: '0-21' },
    { key: 'Sessions per Week', label: 'Sessions per Week', type: 'number', placeholder: 'e.g. 3' },
    { key: 'Engagement Score', label: 'Engagement Score', type: 'number', placeholder: '1-100' },
    { key: 'Mental Health Scale', label: 'Mental Health Scale', type: 'number', placeholder: '1-6' },
    { key: 'Pain Level', label: 'Pain Level', type: 'number', placeholder: '1-10' },
    { key: 'Fatigue Level', label: 'Fatigue Level', type: 'number', placeholder: '1-10' },
    { key: 'Muscle Strength Upper', label: 'Muscle Strength Upper', type: 'number', placeholder: '1-10' },
    { key: 'Muscle Strength Lower', label: 'Muscle Strength Lower', type: 'number', placeholder: '1-10' },
    { key: 'Balance Test Score', label: 'Balance Test Score', type: 'number', placeholder: '1-100' },
    { key: 'Mobility Test Score', label: 'Mobility Test Score', type: 'number', placeholder: '1-100' },
    { key: 'Total Estimated Cost', label: 'Estimated Cost', type: 'number', placeholder: 'e.g. 20000' }
];

export default function FeatureForm({ week, onResult }: FeatureFormProps) {
    // SCIM logic
    const req = requiredScim[week] || [];
    const opt = optionalScim[week] || [];
    const scimFields = [...req, ...opt];

    // Init values
    const initVals: Record<string,string> = {};
    baselineFields.forEach(f => initVals[f.key] = '');
    scimFields.forEach(k => initVals[k] = '');

    const [values, setValues] = useState(initVals);
    const [loading, setLoading] = useState(false);

    const handleChange = (key: string, val: string) => setValues(v => ({...v, [key]:val}));
    const invalid = req.some(k => !values[k]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault(); setLoading(true);
        const payload: Record<string,any> = {};
        baselineFields.forEach(f => { if(values[f.key]) payload[f.key] = f.type==='number'?parseFloat(values[f.key]):values[f.key] });
        scimFields.forEach(k=>{ if(values[k]) payload[k]=parseFloat(values[k]) });
        try{ const p = await getPrediction(week,payload); onResult(p);}catch{}finally{setLoading(false)}
    };

    return (
        <form onSubmit={handleSubmit} className="space-y-8">
            {/* Baseline Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {baselineFields.map(f=>(
                    <div key={f.key} className="flex flex-col">
                        <label className="text-sm font-semibold text-gray-800 mb-1">{f.label}</label>
                        {f.type==='select' && f.options? (
                            <select
                                value={values[f.key]}
                                onChange={e=>handleChange(f.key,e.target.value)}
                                className="border rounded-lg p-2 text-gray-700 focus:ring-blue-500 focus:border-blue-500"
                            >
                                <option value="" disabled>Select {f.label}</option>
                                {f.options.map(o=><option key={o} value={o}>{o}</option>)}
                            </select>
                        ):(
                            <input
                                type={f.type}
                                placeholder={f.placeholder}
                                value={values[f.key]}
                                onChange={e=>handleChange(f.key,e.target.value)}
                                className="border rounded-lg p-2 text-gray-700 focus:ring-blue-500 focus:border-blue-500"
                            />
                        )}
                    </div>
                ))}
            </div>

            {/* SCIM fields */}
            <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                {scimFields.map(k=>(
                    <div key={k} className="flex flex-col">
                        <label className="text-sm font-semibold text-gray-800 mb-1">
                            {k}{req.includes(k)?' *':''}
                        </label>
                        <input
                            type="number"
                            step="0.1"
                            placeholder={req.includes(k)?'Required':'Optional'}
                            value={values[k]}
                            onChange={e=>handleChange(k,e.target.value)}
                            className="border rounded-lg p-2 text-gray-700 focus:ring-blue-500 focus:border-blue-500"
                        />
                    </div>
                ))}
            </div>

            <button
                type="submit"
                disabled={invalid||loading}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg shadow-md transition-colors disabled:opacity-50"
            >
                {loading? 'Predicting...': `Predict Week ${week}`}
            </button>
        </form>
    );
}
