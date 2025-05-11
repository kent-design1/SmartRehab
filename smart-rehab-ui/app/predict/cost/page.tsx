// app/predict/cost/page.tsx
'use client';

import { useSearchParams, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import Link from 'next/link';

interface CostBreakdown {
    baseline_cost: number;
    overuse_cost: number;
    total_cost: number;
    efficiency: number | null;
    explanations: { feature: string; recommendation: string }[];
}

export default function CostOnlyPage() {
    const router = useRouter();
    const params = useSearchParams();

    const costParam = params.get('cost');
    const [cost, setCost] = useState<CostBreakdown | null>(null);
    const [error, setError] = useState('');

    useEffect(() => {
        if (!costParam) {
            setError('No cost parameter in URL.');
            return;
        }
        try {
            const parsed = JSON.parse(decodeURIComponent(costParam));
            setCost(parsed);
        } catch (err) {
            console.error(err);
            setError('Failed to parse cost JSON.');
        }
    }, [costParam]);

    if (error) {
        return (
            <main className="min-h-screen flex items-center justify-center bg-gray-50">
                <div className="text-red-600">{error}</div>
            </main>
        );
    }

    if (!cost) {
        return (
            <main className="min-h-screen flex items-center justify-center bg-gray-50">
                <div>Loading cost…</div>
            </main>
        );
    }

    return (
        <main className="min-h-screen bg-white py-12 px-6">
            <div className="max-w-md mx-auto space-y-6">
                <h1 className="text-2xl font-bold">Cost Breakdown</h1>

                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <dt className="font-medium">Baseline Cost</dt>
                        <dd>CHF {cost.baseline_cost.toLocaleString(undefined, { minimumFractionDigits: 2 })}</dd>
                    </div>
                    <div>
                        <dt className="font-medium">Overuse Cost</dt>
                        <dd>CHF {cost.overuse_cost.toLocaleString(undefined, { minimumFractionDigits: 2 })}</dd>
                    </div>
                    <div>
                        <dt className="font-medium">Total Cost</dt>
                        <dd>CHF {cost.total_cost.toLocaleString(undefined, { minimumFractionDigits: 2 })}</dd>
                    </div>
                    <div>
                        <dt className="font-medium">Efficiency</dt>
                        <dd>
                            {cost.efficiency !== null
                                ? `${cost.efficiency.toFixed(4)} pts/CHF`
                                : 'N/A'}
                        </dd>
                    </div>
                </div>

                <div>
                    <h2 className="text-lg font-semibold mt-6">Cost Explanations</h2>
                    <ul className="list-disc pl-5 mt-2 space-y-1">
                        {cost.explanations.map((ex, i) => (
                            <li key={i} className="text-gray-700">
                                <strong>{ex.feature}:</strong> {ex.recommendation}
                            </li>
                        ))}
                    </ul>
                </div>

                <div className="text-center mt-8">
                    <Link
                        href="/predict"
                        className="inline-block bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
                    >
                        ← Back to Predictor
                    </Link>
                </div>
            </div>
        </main>
    );
}