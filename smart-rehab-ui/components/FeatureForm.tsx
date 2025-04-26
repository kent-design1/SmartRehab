import { useState } from "react";
import { getPrediction } from "@/lib/api";

interface FeatureFormProps {
    week: number;
    onResult: (value: number) => void;
}

const required: Record<number, string[]> = {
    6: ["Total_SCIM_0"],
    12: ["Total_SCIM_0"],
    18: ["Total_SCIM_0"],
    24: ["Total_SCIM_0"],
};
const optional: Record<number, string[]> = {
    6: [],
    12: ["Total_SCIM_6"],
    18: ["Total_SCIM_6", "Total_SCIM_12"],
    24: ["Total_SCIM_6", "Total_SCIM_12", "Total_SCIM_18"],
};

export default function FeatureForm({ week, onResult }: FeatureFormProps) {
    const req = required[week];
    const opt = optional[week];
    const fields = [...req, ...opt];

    const [values, setValues] = useState<Record<string, string>>(
        fields.reduce((acc, f) => ({ ...acc, [f]: "" }), {})
    );
    const [loading, setLoading] = useState(false);

    const handleChange = (field: string) => (e: React.ChangeEvent<HTMLInputElement>) => {
        setValues((v) => ({ ...v, [field]: e.target.value }));
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        try {
            const payload: Record<string, any> = {};
            for (const key of fields) {
                const num = parseFloat(values[key]);
                if (!isNaN(num)) payload[key] = num;
            }
            const pred = await getPrediction(week, payload);
            onResult(pred);
        } catch (err: any) {
            alert(err.message || err);
        } finally {
            setLoading(false);
        }
    };

    const isInvalid = req.some((f) => values[f] === "");

    return (
        <form onSubmit={handleSubmit} className="space-y-4 max-w-md">
            {fields.map((field) => (
                <div key={field}>
                    <label className="block font-medium">
                        {field}
                        {req.includes(field) ? "*" : ""}
                    </label>
                    <input
                        type="number"
                        step="0.1"
                        value={values[field]}
                        onChange={handleChange(field)}
                        className="mt-1 block w-full border rounded p-2"
                    />
                </div>
            ))}
            <button
                type="submit"
                disabled={isInvalid || loading}
                className="w-full bg-blue-500 hover:bg-blue-600 text-white py-2 rounded disabled:opacity-50"
            >
                {loading ? "Predicting…" : "Predict"}
            </button>
        </form> )
}