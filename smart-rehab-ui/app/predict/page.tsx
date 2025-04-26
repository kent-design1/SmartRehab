// app/predict/page.tsx
'use client';
import { useState } from 'react';
import Tabs from '@/components/Tabs';
import FeatureForm from '@/components/FeatureForm';
import PredictionResult from '@/components/PredictionResult';

export default function PredictPage() {
    const weeks = [6, 12, 18, 24];
    const [selected, setSelected] = useState(6);
    const [prediction, setPrediction] = useState<number | null>(null);

    return (
        <section className="min-h-screen  flex items-start justify-center py-16">
            <div className="w-full max-w-3xl bg-white bg-opacity-80 backdrop-blur-sm rounded-3xl shadow-2xl p-10 space-y-10">

                {/* Tabs */}
                <Tabs
                    weeks={weeks}
                    selected={selected}
                    onSelect={(wk) => {
                        setSelected(wk);
                        setPrediction(null);
                    }}
                />

                {/* Form */}
                <div className="px-4 py-6 bg-white rounded-2xl shadow-inner">
                    <FeatureForm week={selected} onResult={setPrediction} />
                </div>

                {/* Result */}
                {prediction !== null && (
                    <div className="transition-transform transform hover:scale-105">
                        <PredictionResult week={selected} value={prediction} />
                    </div>
                )}

            </div>
        </section>
    );
}
