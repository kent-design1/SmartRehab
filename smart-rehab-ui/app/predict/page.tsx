"use client";
import { useState } from "react";
import Tabs from "@/components/Tabs";
import FeatureForm from "@/components/FeatureForm";
import PredictionResult from "@/components/PredictionResult";

export default function PredictPage() {
    const weeks = [6, 12, 18, 24];
    const [selected, setSelected] = useState(6);
    const [prediction, setPrediction] = useState<number | null>(null);

    return (
        <div className="space-y-8">
            <Tabs
                weeks={weeks}
                selected={selected}
                onSelect={(wk) => {
                    setSelected(wk);
                    setPrediction(null);
                }}
            />

            <FeatureForm week={selected} onResult={setPrediction} />

            {prediction !== null && (
                <PredictionResult week={selected} value={prediction} />
            )}
        </div>
    );
}
