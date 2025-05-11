// app/predict/route.ts
'use client';

import { useState } from 'react';
import Tabs from '@/components/Tabs';
import FeatureForm from '@/components/FeatureForm';

export default function PredictPage() {
    const weeks = [6, 12, 18, 24];
    const [selected, setSelected] = useState(6);

    return (
        <section className="min-h-screen flex items-start justify-center py-16 px-8">
            <div className="w-full max-w-3xl space-y-10">
                {/* Week selector */}
                <Tabs
                    weeks={weeks}
                    selected={selected}
                    onSelect={(wk) => {
                        setSelected(wk);
                    }}
                />

                {/* Form only — it now pushes you to /predict/results when you submit */}
                <div className="px-6 py-8 bg-white rounded-2xl shadow">
                    <FeatureForm week={selected} />
                </div>
            </div>
        </section>
    );
}