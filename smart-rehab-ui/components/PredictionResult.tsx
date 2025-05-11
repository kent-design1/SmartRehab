
import { Activity } from 'lucide-react';

interface PredictionResultProps {
    week: number;
    value: number;
}

export default function PredictionResult({ week, value }: PredictionResultProps) {
    return (
        <div className="max-w-sm mx-auto p-6 bg-gradient-to-br from-blue-50 to-blue-100 rounded-2xl shadow-lg border border-blue-200">
            <div className="flex justify-center mb-4">
                <Activity className="w-10 h-10 text-blue-500" />
            </div>
            <h2 className="text-center text-xl font-semibold text-blue-700">
                Predicted SCIM at Week {week}
            </h2>
            <p className="text-center text-5xl font-extrabold text-gray-900 mt-2">
                {value.toFixed(2)}
            </p>
            <p className="text-center text-sm text-gray-500 mt-1">
                (Spinal Cord Independence Measure)
            </p>
        </div>
    );
}