'use client';

import { useSearchParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { useMemo } from 'react';
import PredictionResult from '@/components/PredictionResult';
import { ArrowLeft } from 'lucide-react';
import {
    ResponsiveContainer,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    CartesianGrid,
    PieChart,
    Pie,
    Cell,
} from 'recharts';
import {
    CostBreakdown,
    StaticRecommendation,
    ShapRecommendation,
} from '@/lib/api';

const COLORS = ['#3b82f6', '#6366f1', '#10b981', '#f59e0b'];

export default function ResultsPage() {
    const router = useRouter();
    const params = useSearchParams();

    const weekParam = params.get('week');
    const predParam = params.get('prediction');
    const costParam = params.get('cost');
    const staticParam = params.get('static_recs');
    const shapParam = params.get('shap_recs');

    // we only *require* week, prediction and cost
    if (!weekParam || !predParam || !costParam) {
        return (
            <main className="min-h-screen flex items-center justify-center bg-gray-100">
                <div className="text-center space-y-4">
                    <p className="text-lg text-gray-600">Oops — missing prediction data.</p>
                    <Link href="/predict" className="inline-flex items-center text-blue-600 hover:underline">
                        <ArrowLeft className="w-5 h-5 mr-2" /> Back to Predictor
                    </Link>
                </div>
            </main>
        );
    }

    const week = parseInt(weekParam, 10);
    const prediction = parseFloat(predParam);
    const cost: CostBreakdown = JSON.parse(decodeURIComponent(costParam));

    // fall back to [] if missing or literally "undefined"
    const staticRecs: StaticRecommendation[] = JSON.parse(
        staticParam && staticParam !== 'undefined'
            ? decodeURIComponent(staticParam)
            : '[]'
    );
    const shapRecs: ShapRecommendation[] = JSON.parse(
        shapParam && shapParam !== 'undefined'
            ? decodeURIComponent(shapParam)
            : '[]'
    );

    // build SHAP bar & pie data
    const barData = useMemo(
        () =>
            shapRecs
                .filter((r) => r.shap_value != null)
                .map((r) => ({
                    feature: r.feature.replace(/^(num__|cat__)/, ''),
                    shap: r.shap_value,
                }))
                .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
                .slice(0, 5),
        [shapRecs]
    );
    const pieData = useMemo(
        () => barData.map((d) => ({ name: d.feature, value: Math.abs(d.shap) })),
        [barData]
    );

    // session & therapy from SHAP
    const sessionRec = shapRecs.find((r) => r.feature === 'SessionsPerWeek');
    const therapyRec = shapRecs.find((r) => r.feature === 'TherapyPlanRecommendations');

    return (
        <main className="min-h-screen bg-gradient-to-br from-indigo-50 to-blue-100 py-16 px-6 lg:px-24">
            <div className="mx-auto max-w-4xl space-y-12">
                {/* Header */}
                <div className="flex items-center space-x-4">
                    <button onClick={() => router.back()} className="p-2 rounded-full hover:bg-gray-200">
                        <ArrowLeft className="w-6 h-6 text-gray-700" />
                    </button>
                    <h1 className="text-4xl font-extrabold text-gray-900">Week {week} Prediction</h1>
                </div>

                {/* Prediction */}
                <PredictionResult week={week} value={prediction} />

                {/* Quick tips (static rules) */}
                {staticRecs.length > 0 && (
                    <section className="space-y-6">
                        <h2 className="text-2xl font-semibold">Quick Tips</h2>
                        {staticRecs.map((r, i) => (
                            <div key={i} className="p-4 bg-white rounded-lg shadow">
                                <h3 className="font-medium">{r.feature}</h3>
                                <p className="italic text-gray-600">{r.rationale}</p>
                                <p className="mt-1">{r.recommendation}</p>
                            </div>
                        ))}
                    </section>
                )}

                {/* Session & Therapy suggestions from SHAP */}
                {sessionRec && (
                    <div className="rounded-2xl border-l-4 border-yellow-500 bg-yellow-100 p-6">
                        <h3 className="font-semibold text-yellow-800">Session Adjustment</h3>
                        <p className="text-gray-700">{sessionRec.recommendation}</p>
                    </div>
                )}
                {therapyRec && (
                    <div className="rounded-2xl border-l-4 border-green-500 bg-green-50 p-6">
                        <h3 className="font-semibold text-green-800">Next Therapy Plan</h3>
                        <p className="text-gray-700">{therapyRec.recommendation}</p>
                    </div>
                )}

                {/* Cost & Efficiency */}
                <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="rounded-2xl bg-white p-6 shadow-lg">
                        <h3 className="text-lg font-semibold text-gray-800">Baseline Cost</h3>
                        <p className="mt-2 text-2xl font-bold">CHF {cost.baseline_cost.toLocaleString()}</p>
                    </div>
                    <div className="rounded-2xl bg-white p-6 shadow-lg">
                        <h3 className="text-lg font-semibold text-gray-800">Overuse Cost</h3>
                        <p className="mt-2 text-2xl font-bold">CHF {cost.overuse_cost.toLocaleString()}</p>
                    </div>
                    <div className="rounded-2xl bg-white p-6 shadow-lg">
                        <h3 className="text-lg font-semibold text-gray-800">Total & Efficiency</h3>
                        <p className="mt-2 text-2xl font-bold">CHF {cost.total_cost.toLocaleString()}</p>
                        <p className="mt-1 text-sm">
                            {cost.efficiency != null ? `${cost.efficiency.toFixed(4)} pts/CHF` : 'N/A'}
                        </p>
                    </div>
                </section>

                {/* Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Donut */}
                    <section className="rounded-2xl bg-white p-6 shadow-lg">
                        <h2 className="text-2xl mb-4">Top Drivers (Donut)</h2>
                        <ResponsiveContainer width="100%" height={280}>
                            <PieChart>
                                <Pie
                                    data={pieData}
                                    dataKey="value"
                                    nameKey="name"
                                    innerRadius="60%"
                                    outerRadius="80%"
                                >
                                    {pieData.map((_, idx) => (
                                        <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip formatter={(v: number) => v.toFixed(2)} />
                            </PieChart>
                        </ResponsiveContainer>
                    </section>

                    {/* Signed Bar */}
                    <section className="rounded-2xl bg-white p-6 shadow-lg">
                        <h2 className="text-2xl mb-4">Feature Importance (SHAP)</h2>
                        <ResponsiveContainer width="100%" height={280}>
                            <BarChart data={barData} layout="vertical" margin={{ left: 100 }}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                <XAxis type="number" tickFormatter={(v) => v.toFixed(1)} />
                                <YAxis dataKey="feature" type="category" width={120} />
                                <Tooltip formatter={(v: number) => v.toFixed(2)} />
                                <Bar dataKey="shap" barSize={16}>
                                    {barData.map((entry, idx) => (
                                        <Cell
                                            key={idx}
                                            fill={entry.shap >= 0 ? '#10b981' : '#ef4444'}
                                        />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </section>
                </div>

                {/* Detailed SHAP Recommendations */}
                {shapRecs.length > 0 && (
                    <section className="bg-white p-6 rounded-2xl shadow-lg">
                        <h2 className="text-2xl mb-4">Detailed Recommendations</h2>
                        <ul className="space-y-3">
                            {shapRecs.map((r, i) => (
                                <li key={i} className="border-l-4 border-blue-500 bg-gray-50 p-4 rounded-lg">
                                    <div className="flex justify-between">
                                        <span className="font-medium">{r.feature.replace(/^(num__|cat__)/, '')}</span>
                                        <span className="italic">{r.shap_value.toFixed(2)}</span>
                                    </div>
                                    <p className="mt-1">{r.recommendation}</p>
                                    <p className="mt-1 italic text-gray-600">{r.rationale}</p>
                                </li>
                            ))}
                        </ul>
                    </section>
                )}

                <div className="text-center">
                    <Link href="/predict" className="inline-block bg-blue-600 text-white px-8 py-3 rounded-full">
                        Make another prediction
                    </Link>
                </div>
            </div>
        </main>
    );
}
