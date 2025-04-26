// app/predict/results/page.tsx
'use client';

import { useSearchParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import PredictionResult from '@/components/PredictionResult';
import { ArrowLeft } from 'lucide-react';

export default function ResultsPage() {
    const params = useSearchParams();
    const router = useRouter();

    // Retrieve query params
    const weekParam = params.get('week');
    const predParam = params.get('prediction');

    // Validate
    if (!weekParam || !predParam) {
        return (
            <main className="min-h-screen bg-gray-100 flex items-center justify-center">
                <div className="text-center space-y-4">
                    <p className="text-lg text-gray-700">No prediction data found.</p>
                    <Link
                        href="/predict"
                        className="inline-flex items-center text-blue-600 hover:underline"
                    >
                        <ArrowLeft className="w-5 h-5 mr-2" /> Back to Predictor
                    </Link>
                </div>
            </main>
        );
    }

    const week = parseInt(weekParam, 10);
    const prediction = parseFloat(predParam);

    return (
        <main className="min-h-screen bg-gradient-to-br from-indigo-50 to-blue-100 py-16 px-6 lg:px-24">
            <div className="max-w-lg mx-auto bg-white rounded-3xl shadow-2xl p-10 space-y-8">
                <button
                    onClick={() => router.back()}
                    className="flex items-center text-gray-600 hover:text-gray-800 transition"
                >
                    <ArrowLeft className="w-5 h-5 mr-2" /> Back
                </button>

                <h1 className="text-3xl font-extrabold text-gray-900 text-center">
                    Week {week} Prediction
                </h1>

                <div className="flex justify-center">
                    <PredictionResult week={week} value={prediction} />
                </div>

                <div className="text-center">
                    <Link
                        href="/predict"
                        className="text-blue-600 hover:underline"
                    >
                        Make another prediction →
                    </Link>
                </div>
            </div>
        </main>
    );
}
